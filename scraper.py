"""
scraper.py - Fetch up to 10 documents of one type for one matter from the UARB site.

    result = fetch_documents("M12205", "Other Documents")

fetch_documents() NEVER raises. It always returns a ScrapeResult. If result.ok is False,
result.user_message is safe to email to the sender (details go to logs only).
"ok=True with no files" is a normal outcome: the matter exists but that tab is empty.

Environment variables (.env), all optional:
  BROWSER    chrome (default) | msedge | chromium | firefox | webkit
  HEADLESS   false (default while developing) | true
"""
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

load_dotenv()
log = logging.getLogger(__name__)

SITE_URL = "https://uarb.novascotia.ca/fmi/webd/UARB15"
DOC_TYPES = ["Exhibits", "Key Documents", "Other Documents", "Transcripts", "Recordings"]
MAX_DOCS = 10

# From your codegen recording. FileMaker generates these IDs, so if the site is ever
# updated this is the first thing that could break. (The "eg M01234" hint is just a <div>, not a
# real placeholder attribute, so it can't be used to find the field.)
MATTER_INPUT_SELECTOR = "#b0p0o254i0i0r1 > .inner_border > .text"

PAGE_LOAD_TIMEOUT_MS = 45_000
DEFAULT_TIMEOUT_MS = 30_000
DOWNLOAD_TIMEOUT_MS = 60_000
DELAY_BETWEEN_DOWNLOADS_S = 0.5  # be polite to the server
RUN_BUDGET_S = 300  # stop downloading after 5 minutes total
SCREENSHOT_DIR = Path("screenshots")

# After clicking GO GET IT, the site shows a button named after the file (e.g. "102674.pdf").
FILE_BUTTON_RE = re.compile(r"^[\w\-. ()]+\.[A-Za-z0-9]{2,5}$")
GO_GET_IT_RE = re.compile(r"go get it", re.IGNORECASE)

SITE_DOWN_MSG = ("The UARB website isn't responding properly right now. "
                 "Please try again in a little while.")
SYSTEM_ERROR_MSG = ("Sorry, I ran into a technical problem while fetching your documents. "
                    "Please try again in a few minutes.")
RETRYABLE_CODES = {"SITE_UNREACHABLE", "SITE_TIMEOUT"}


@dataclass
class ScrapeResult:
    ok: bool
    matter_number: Optional[str] = None
    document_type: Optional[str] = None
    type_counts: dict = field(default_factory=dict)  # {"Exhibits": 13, ...}; None = unreadable
    files: list = field(default_factory=list)  # Paths of downloaded files
    failed_files: list = field(default_factory=list)  # human-readable reasons for skipped rows
    header_text: str = ""  # raw text from the top of the matter page (for the summary LLM)
    work_dir: Optional[Path] = None  # delete this after you've sent the email
    error_code: Optional[str] = None
    user_message: Optional[str] = None  # safe to email to the sender
    detail: Optional[str] = None  # logs only
    notify_developer: bool = False

    @property
    def total_files(self) -> int:
        return sum(v for v in self.type_counts.values() if isinstance(v, int))


class ScraperError(Exception):
    def __init__(self, code, user_message, detail=None, dev=False):
        super().__init__(f"{code}: {detail or user_message}")
        self.code, self.user_message, self.detail, self.dev = code, user_message, detail, dev


class _RowError(Exception):
    """One row failed to download (not fatal for the whole run)."""


# ---------------------------------------------------------------- browser
def _launch(p, name: str, headless: bool):
    name = name.lower()
    if name in ("chrome", "msedge", "edge"):
        channel = "msedge" if name in ("msedge", "edge") else "chrome"
        return p.chromium.launch(channel=channel, headless=headless)
    if name == "chromium":
        return p.chromium.launch(headless=headless)
    if name == "firefox":
        return p.firefox.launch(headless=headless)
    if name in ("webkit", "safari"):
        return p.webkit.launch(headless=headless)
    raise PlaywrightError(f"Unknown BROWSER value: {name!r}")


def _open_browser(p):
    """Try the configured browser first, then fall back to others that are installed."""
    configured = os.getenv("BROWSER", "chrome").strip().lower()
    headless = os.getenv("HEADLESS", "false").strip().lower() in ("1", "true", "yes")
    order = [configured] + [b for b in ("chrome", "msedge", "chromium", "firefox") if b != configured]
    errors = []
    for name in order:
        try:
            browser = _launch(p, name, headless)
            if name != configured:
                log.warning("Browser %r unavailable, using %r instead", configured, name)
            return browser
        except PlaywrightError as exc:
            errors.append(f"{name}: {str(exc).splitlines()[0]}")
    raise ScraperError("BROWSER_UNAVAILABLE", SYSTEM_ERROR_MSG,
                       "No usable browser. " + " | ".join(errors), dev=True)


def _screenshot(page, label: str):
    try:
        SCREENSHOT_DIR.mkdir(exist_ok=True)
        path = SCREENSHOT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{label}.png"
        page.screenshot(path=str(path))
        log.info("Saved failure screenshot: %s", path)
    except Exception:  # never let a screenshot problem hide the real error
        pass


# ---------------------------------------------------------------- page helpers
def _tab(page, doc_type: str):
    """Tab buttons are labelled like 'Other Documents - 21'."""
    return page.get_by_role(
        "button", name=re.compile(rf"^\s*{re.escape(doc_type)}\s*[-\u2013\u2014]\s*\d+", re.I))


def _read_counts(page) -> dict:
    counts = {}
    for t in DOC_TYPES:
        try:
            text = _tab(page, t).first.inner_text(timeout=3000)
            m = re.search(r"[-\u2013\u2014]\s*(\d+)", text)
            counts[t] = int(m.group(1)) if m else None
        except PlaywrightError:
            counts[t] = None
    return counts


def _read_box(page, box) -> str:
    """Best-effort read of what's currently typed in the matter box. Returns '' if unreadable."""
    readers = (
        lambda: box.inner_text(timeout=1000),
        lambda: box.locator("input, textarea").first.input_value(timeout=1000),
        lambda: page.evaluate("() => { const e = document.activeElement;"
                              " return (e && (e.value || e.innerText)) || ''; }"),
    )
    for read in readers:
        try:
            text = (read() or "").strip()
            if text and len(text) <= 30:  # a field's content is short; ignore whole-page text
                return text
        except PlaywrightError:
            pass
    return ""


def _enter_matter(page, box, matter: str):
    """Type the matter number into the FileMaker box and make sure it all arrived.

    FileMaker needs a moment after the click before the field accepts keys, and can swallow the
    first characters (we saw 'M12205' arrive as '2205'). So: wait, type, look at what's in the box,
    and if it's wrong clear it and try again."""
    shown = ""
    for attempt in (1, 2, 3):
        box.click()
        page.wait_for_timeout(700)
        if attempt > 1:  # clear the leftovers from the previous attempt
            page.keyboard.press("Control+A")
            page.keyboard.press("Backspace")
            page.wait_for_timeout(200)
        page.keyboard.type(matter, delay=90)
        page.wait_for_timeout(300)
        shown = _read_box(page, box)
        # Only retry when we can SEE wrong text. If the box can't be read, trust the typing.
        if not shown or matter.lower() in re.sub(r"\s", "", shown).lower():
            return
        log.warning("Matter box shows %r instead of %r (attempt %d), retrying", shown, matter, attempt)
    raise ScraperError("SEARCH_BOX_NOT_FILLED", SYSTEM_ERROR_MSG,
                       f"box kept showing {shown!r} instead of {matter!r}", dev=True)


def _click_search(page, box):
    """Click the Search button that sits next to the 'Go Directly to Matter' box.

    The page has several 'Search' buttons, and your recording used .nth(4), which is fragile.
    Instead pick the one closest to the input box."""
    try:
        box_bb = box.bounding_box(timeout=2000)
    except PlaywrightError:
        box_bb = None
    if not box_bb:
        return  # box is gone, so the earlier Enter press already ran the search

    buttons = page.get_by_role("button", name="Search", exact=True)
    if buttons.count() == 0:
        buttons = page.get_by_role("button", name="Search")
    best, best_dist = None, None
    for i in range(buttons.count()):
        bb = buttons.nth(i).bounding_box()
        if not bb:
            continue
        dist = (bb["x"] - box_bb["x"]) ** 2 + (bb["y"] - box_bb["y"]) ** 2
        if best_dist is None or dist < best_dist:
            best, best_dist = buttons.nth(i), dist
    (best or buttons.first).click()


def _wait_for_matter_page(page, timeout_ms: int) -> bool:
    try:
        _tab(page, "Exhibits").first.wait_for(state="visible", timeout=timeout_ms)
        return True
    except PlaywrightTimeout:
        return False


def _ensure_rows(page, buttons, needed: int) -> int:
    """The document list may only render some rows until you scroll. Scroll until we have
    `needed` rows, or the count stops growing. Returns the row count we ended up with."""
    last, stalls = buttons.count(), 0
    while last < needed and stalls < 3:
        try:
            buttons.last.hover(timeout=3000)
            page.mouse.wheel(0, 800)
        except PlaywrightError:
            pass
        page.wait_for_timeout(600)
        now = buttons.count()
        stalls = stalls + 1 if now == last else 0
        last = now
    return last


def _clear_popup(page):
    """Close the 'Download Files' dialog if it's showing.

    The site leaves this dialog open after a download, and its overlay blocks clicks on the
    page behind it, so it must be closed before the next row can be downloaded."""
    file_btn = page.get_by_role("button", name=FILE_BUTTON_RE).first
    if not file_btn.is_visible():
        return  # no dialog showing, nothing to do

    close_btn = page.get_by_role("button", name="Close", exact=True).first
    try:
        close_btn.click(timeout=3000)
    except PlaywrightError:
        log.warning("Couldn't click Close on the download dialog; trying Escape")
        page.keyboard.press("Escape")
    file_btn.wait_for(state="hidden", timeout=5000)  # raises if the dialog never goes away


def _safe_clear_popup(page):
    """Best-effort cleanup for error paths: never raises."""
    try:
        _clear_popup(page)
    except PlaywrightError:
        pass


def _safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(name).name).strip() or "document"


def _unique_path(folder: Path, name: str) -> Path:
    path = folder / _safe_name(name)
    stem, suffix, n = path.stem, path.suffix, 1
    while path.exists():
        path = folder / f"{stem}_{n}{suffix}"
        n += 1
    return path


def _download_row(page, buttons, i: int, dest: Path, seen: set) -> Path:
    """GO GET IT on row i -> filename button appears -> click it -> save the download."""
    _clear_popup(page)
    btn = buttons.nth(i)
    btn.scroll_into_view_if_needed()
    btn.click()

    file_btn = page.get_by_role("button", name=FILE_BUTTON_RE).first
    file_btn.wait_for(state="visible", timeout=15_000)
    with page.expect_download(timeout=DOWNLOAD_TIMEOUT_MS) as info:
        file_btn.click()
    download = info.value

    name = download.suggested_filename
    if name in seen:
        raise _RowError(f"duplicate of {name} (previous popup may not have closed)")
    target = _unique_path(dest, name)
    download.save_as(target)  # waits for completion, raises if the download failed
    if not target.exists() or target.stat().st_size == 0:
        raise _RowError(f"{name} downloaded empty")
    seen.add(name)
    _clear_popup(page)
    return target


# ---------------------------------------------------------------- the main flow
def _scrape(page, matter: str, doc_type: str, dest: Path, max_docs: int) -> ScrapeResult:
    started = time.monotonic()

    # 1. Open the site
    try:
        response = page.goto(SITE_URL, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT_MS)
    except PlaywrightTimeout as exc:
        raise ScraperError("SITE_TIMEOUT", SITE_DOWN_MSG, f"goto timed out: {exc}")
    except PlaywrightError as exc:
        raise ScraperError("SITE_UNREACHABLE", SITE_DOWN_MSG, f"goto failed: {exc}")
    if response is not None and response.status >= 500:
        raise ScraperError("SITE_UNREACHABLE", SITE_DOWN_MSG, f"HTTP {response.status}")

    # 2. Enter the matter number in "Go Directly to Matter" and search
    box = page.locator(MATTER_INPUT_SELECTOR).first
    try:
        box.wait_for(state="visible", timeout=PAGE_LOAD_TIMEOUT_MS)
    except PlaywrightTimeout:
        raise ScraperError("SEARCH_BOX_NOT_FOUND", SITE_DOWN_MSG,
                           "Matter input never appeared (site slow, or the layout/IDs changed)",
                           dev=True)
    # This "field" is a FileMaker-drawn <div>, not an <input>, so box.fill() is refused.
    # Click to focus it and type with real key presses (already normalized to MXXXXX by the parser).
    _enter_matter(page, box, matter)
    page.keyboard.press("Enter")  # your recording ended the text with a newline; commits the field
    _click_search(page, box)

    # 3. Wait for the matter page (the tab buttons). Retry the click once if we're still on the form.
    found = _wait_for_matter_page(page, 12_000)
    if not found and box.is_visible():
        log.info("Still on the search form; clicking Search again")
        _click_search(page, box)
        found = _wait_for_matter_page(page, 15_000)
    if not found:
        if box.is_visible():
            raise ScraperError(
                "MATTER_NOT_FOUND",
                f"I couldn't find matter {matter} on the UARB website. "
                "Please double-check the matter number and try again.",
                "Search form still showing after search: matter missing, or the click didn't register",
                dev=False)
        raise ScraperError("SITE_TIMEOUT", SITE_DOWN_MSG, "Matter page never finished loading")

    # 4. Counts for every tab + raw header text (title, status, dates...) for the summary
    counts = _read_counts(page)
    try:
        header_text = page.locator("body").inner_text(timeout=5000)[:2500]
    except PlaywrightError:
        header_text = ""
    if header_text and matter not in header_text:
        log.warning("Matter number %s not found in page text; is this the right matter?", matter)
    log.info("Tab counts for %s: %s", matter, counts)

    result = ScrapeResult(ok=True, matter_number=matter, document_type=doc_type,
                          type_counts=counts, header_text=header_text, work_dir=dest)

    total = counts.get(doc_type)
    if total == 0:
        return result  # valid outcome: nothing to download

    # 5. Open the requested tab
    try:
        _tab(page, doc_type).first.click()
        page.get_by_role("button", name=GO_GET_IT_RE).first.wait_for(state="visible", timeout=20_000)
    except PlaywrightTimeout:
        if total is None:  # couldn't read a count and there are no rows -> treat as empty
            return result
        raise ScraperError("TAB_NOT_FOUND", SYSTEM_ERROR_MSG,
                           f"Tab {doc_type!r} (count {total}) showed no GO GET IT buttons", dev=True)

    # 6. Download up to max_docs. Failed rows are skipped and the next row is tried instead.
    buttons = page.get_by_role("button", name=GO_GET_IT_RE)
    seen, row, consecutive_failures = set(), 0, 0
    max_rows = min(total, max_docs * 2) if total is not None else max_docs * 2
    while len(result.files) < max_docs and row < max_rows:
        if time.monotonic() - started > RUN_BUDGET_S:
            result.failed_files.append("stopped early: time budget reached")
            break
        if consecutive_failures >= 3:
            result.failed_files.append("stopped early: 3 failures in a row")
            break
        i, row = row, row + 1
        if _ensure_rows(page, buttons, i + 1) <= i:
            break  # the list has fewer rows than the tab count suggested

        last_error = None
        for attempt in (1, 2):
            try:
                result.files.append(_download_row(page, buttons, i, dest, seen))
                last_error = None
                break
            except _RowError as exc:
                last_error = str(exc)
                break  # retrying won't help
            except PlaywrightError as exc:
                last_error = str(exc).splitlines()[0]
                log.warning("Row %d attempt %d failed: %s", i + 1, attempt, last_error)
                _safe_clear_popup(page)
        _safe_clear_popup(page)  # e.g. after a duplicate/empty file the dialog is still open
        if last_error:
            result.failed_files.append(f"row {i + 1}: {last_error}")
            consecutive_failures += 1
        else:
            consecutive_failures = 0
        time.sleep(DELAY_BETWEEN_DOWNLOADS_S)

    if not result.files:
        raise ScraperError(
            "DOWNLOAD_FAILED",
            f"I found matter {matter} with {total if total is not None else 'some'} {doc_type}, "
            "but I wasn't able to download any of them. Please try again later.",
            f"All downloads failed: {result.failed_files}", dev=True)
    return result


def _run_once(matter: str, doc_type: str, dest: Path, max_docs: int) -> ScrapeResult:
    with sync_playwright() as p:
        browser = _open_browser(p)
        try:
            context = browser.new_context(accept_downloads=True,
                                          viewport={"width": 1400, "height": 1200})
            page = context.new_page()
            page.set_default_timeout(DEFAULT_TIMEOUT_MS)
            try:
                return _scrape(page, matter, doc_type, dest, max_docs)
            except ScraperError as exc:
                _screenshot(page, exc.code)
                raise
            except PlaywrightTimeout as exc:
                _screenshot(page, "TIMEOUT")
                raise ScraperError("SITE_TIMEOUT", SITE_DOWN_MSG, str(exc))
            except PlaywrightError as exc:
                _screenshot(page, "PLAYWRIGHT_ERROR")
                raise ScraperError("SITE_ERROR", SYSTEM_ERROR_MSG, str(exc), dev=True)
        finally:
            browser.close()


def _failed(code, user_message, detail, dev, matter, doc_type, work_dir) -> ScrapeResult:
    return ScrapeResult(ok=False, matter_number=matter, document_type=doc_type, work_dir=work_dir,
                        error_code=code, user_message=user_message, detail=detail,
                        notify_developer=dev)


def fetch_documents(matter_number: str, document_type: str, dest_dir=None,
                    max_docs: int = MAX_DOCS) -> ScrapeResult:
    """Never raises. Check result.ok. Delete result.work_dir when you're done with the files."""
    work_dir = None
    try:
        # Defense in depth: the parser validates too, but this function drives a real website.
        if not re.fullmatch(r"M\d{5}", matter_number or ""):
            return _failed("INVALID_INPUT", SYSTEM_ERROR_MSG, f"bad matter {matter_number!r}",
                           True, matter_number, document_type, None)
        if document_type not in DOC_TYPES:
            return _failed("INVALID_INPUT", SYSTEM_ERROR_MSG, f"bad type {document_type!r}",
                           True, matter_number, document_type, None)

        work_dir = Path(dest_dir) if dest_dir else Path(
            tempfile.mkdtemp(prefix=f"uarb_{matter_number}_"))
        work_dir.mkdir(parents=True, exist_ok=True)

        for attempt in (1, 2):
            try:
                result = _run_once(matter_number, document_type, work_dir, max_docs)
                log.info("Scrape ok: %d file(s), %d skipped", len(result.files),
                         len(result.failed_files))
                return result
            except ScraperError as exc:
                if exc.code in RETRYABLE_CODES and attempt == 1:
                    log.warning("Transient failure (%s), retrying once", exc.code)
                    shutil.rmtree(work_dir, ignore_errors=True)
                    work_dir.mkdir(parents=True, exist_ok=True)
                    time.sleep(3)
                    continue
                log.warning("Scrape failed [%s] %s", exc.code, exc.detail or "")
                return _failed(exc.code, exc.user_message, exc.detail, exc.dev,
                               matter_number, document_type, work_dir)
    except Exception as exc:  # last-resort safety net
        log.exception("Unexpected error while scraping")
        return _failed("UNEXPECTED_ERROR", SYSTEM_ERROR_MSG, repr(exc), True,
                       matter_number, document_type, work_dir)


if __name__ == "__main__":
    # Manual test:  python scraper.py M12205 "Other Documents"
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    m = sys.argv[1] if len(sys.argv) > 1 else "M12205"
    t = sys.argv[2] if len(sys.argv) > 2 else "Other Documents"
    r = fetch_documents(m, t)
    print("\nok:", r.ok, "| code:", r.error_code)
    print("counts:", r.type_counts, "| total:", r.total_files)
    print("files:", [f.name for f in r.files])
    print("skipped:", r.failed_files)
    print("folder:", r.work_dir)
    if r.ok:
        print("\n--- header text (check this looks like the matter details) ---")
        print(r.header_text[:800])
    else:
        print("reply to user:", r.user_message)