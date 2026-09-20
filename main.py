"""
main.py - The document-fetching agent. Start it and leave it running:

    python main.py          # poll the inbox forever (Ctrl+C to stop)
    python main.py --once   # handle whatever is unread right now, then exit (good for testing)

Pipeline for each unread email:
    inbox.py    read it (and ignore automated mail / the agent's own mail)
    llm.py      work out the matter number + document type          -> parse_request()
    scraper.py  download up to 10 documents from the UARB site      -> fetch_documents()
    mailer.py   ZIP them, reply with the ZIP attached               -> make_zip(), send_email()
    summary.py  metadata + counts written up in the reply           -> extract_metadata(), build_email_body()

Every email gets a reply, even when something fails. Requests are handled one at a time.

Optional environment variables (.env):
    POLL_INTERVAL_S          seconds between inbox checks (default 30)
    MAX_REQUESTS_PER_HOUR    per-sender limit, protects your API bill (default 20)
    ALLOWED_SENDERS          comma-separated addresses; if set, everyone else is ignored (default: anyone)
"""
import imaplib
import json
import logging
import logging.handlers
import os
import re
import shutil
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from inbox import InboxError, check_login, fetch_unread, mark_seen  # noqa: E402
from llm import parse_request  # noqa: E402
from mailer import MAX_ZIP_BYTES, make_zip, send_email  # noqa: E402
from scraper import fetch_documents  # noqa: E402
from summary import (build_email_body, build_plain_body, extract_metadata,  # noqa: E402
                     greeting_name)

log = logging.getLogger("agent")

POLL_INTERVAL_S = int(os.getenv("POLL_INTERVAL_S", "30"))
MAX_REQUESTS_PER_SENDER_PER_HOUR = int(os.getenv("MAX_REQUESTS_PER_HOUR", "20"))
MAX_ATTEMPTS = 3  # if a reply can't be sent, retry on later polls, then give up
PROCESSED_FILE = Path("processed.json")
LOG_FILE = Path("agent.log")
# Sending problems where retrying can't help.
PERMANENT_SEND_ERRORS = {"INVALID_RECIPIENT", "RECIPIENT_REFUSED", "ATTACHMENT_TOO_LARGE", "MESSAGE_REJECTED"}
# If a big ZIP can't be sent (slow or blocked upload), try smaller ones, then a text-only summary.
FALLBACK_ZIP_BUDGETS = [MAX_ZIP_BYTES, 6 * 1024 * 1024, 2 * 1024 * 1024]
GENERIC_ERROR = ("Sorry, something went wrong on my end while handling your request. "
                 "Please try again in a few minutes.")


# ================================================================ small helpers
class ProcessedStore:
    """Remembers which emails were already answered, so a restart never sends duplicate replies."""

    def __init__(self, path: Path, keep: int = 2000):
        self.path, self.keep = Path(path), keep
        try:
            self.keys = [k for k in json.loads(self.path.read_text("utf-8")) if isinstance(k, str)]
        except (OSError, ValueError):
            self.keys = []
        self._seen = set(self.keys)

    def __contains__(self, key: str) -> bool:
        return key in self._seen

    def add(self, key: str):
        if key in self._seen:
            return
        self.keys.append(key)
        self._seen.add(key)
        if len(self.keys) > self.keep:
            self._seen.discard(self.keys.pop(0))
        try:  # write-then-replace so a crash can't leave a half-written file
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.keys), "utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("Couldn't save %s (%r); continuing with the in-memory list", self.path, exc)


class RateLimiter:
    def __init__(self, limit: int, window_s: int = 3600):
        self.limit, self.window_s, self.hits = limit, window_s, defaultdict(deque)

    def allow(self, who: str) -> bool:
        now, q = time.time(), self.hits[who]
        while q and now - q[0] > self.window_s:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


def _reply_subject(subject: str) -> str:
    s = re.sub(r"\s+", " ", subject or "").strip() or "your request"
    return s if s.lower().startswith("re:") else f"Re: {s}"


def keep_awake():
    """Ask Windows not to go to sleep while the agent runs (closing the lid can still sleep it,
    depending on your power settings)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        log.info("Asked Windows to stay awake while the agent runs")
    except Exception as exc:
        log.warning("Couldn't set keep-awake: %r", exc)


def setup_logging():
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logfile = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3,
                                                   encoding="utf-8")
    logfile.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(logfile)


def check_config():
    needed = ["GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "OPENAI_API_KEY", "OPENAI_MODEL"]
    missing = [v for v in needed if not os.getenv(v, "").strip()]
    if missing:
        sys.exit("Missing settings in your .env file: " + ", ".join(missing))


# ================================================================ handling one email
def deliver(result, metadata, name, send):
    """Build the success email and send it. Returns a SendResult.

    If a big ZIP can't be sent, fall back to smaller ZIPs, and finally to a text-only summary,
    so the sender always gets an answer."""
    failed = sum(1 for f in result.failed_files if f.startswith("row "))

    def body_for(too_large, downloaded):
        return build_email_body(
            result.matter_number, result.document_type, result.type_counts, downloaded=downloaded,
            metadata=metadata, failed=failed, too_large=too_large, recipient_name=name)

    if not result.files:  # empty tab: nothing to attach
        return send(body_for([], 0))

    last_included = None
    for budget in FALLBACK_ZIP_BUDGETS:
        zr = make_zip(result.files, result.work_dir, result.matter_number, result.document_type,
                      max_bytes=budget)
        if not zr.ok:  # nothing fits under this budget (or the ZIP itself failed)
            if zr.error_code != "ZIP_TOO_LARGE":
                log.error("NEEDS ATTENTION: zip [%s] %s", zr.error_code, zr.detail)
            return send(body_for(zr.skipped_too_large, 0))
        included = tuple(f.name for f in zr.included)
        if included == last_included:
            continue  # identical to the ZIP that just failed to send; don't repeat it
        last_included = included
        sr = send(body_for(zr.skipped_too_large, len(zr.included)), zr.zip_path)
        if sr.ok or sr.error_code != "SEND_FAILED":
            return sr
        log.warning("Couldn't send a %.1f MB ZIP; trying a smaller one", zr.size_bytes / 1e6)

    # Even the smallest ZIP couldn't be sent: send the summary without an attachment.
    everything = [(f.name, f.stat().st_size) for f in result.files if f.exists()]
    log.warning("Sending the summary without an attachment")
    return send(body_for(everything, 0))


def handle_message(em) -> bool:
    """Do the whole job for one email and reply to the sender.
    Returns True when finished (mark it read), False to leave it unread and retry on the next poll."""
    subject = _reply_subject(em.subject)
    references = " ".join(x for x in [em.references, em.message_id] if x) or None
    name = greeting_name(em.from_header)
    replied = {"ok": False}

    def send(body: str, attachment=None):
        sr = send_email(em.from_addr, subject, body, attachment_path=attachment,
                        in_reply_to=em.message_id, references=references)
        if sr.ok:
            replied["ok"] = True
        elif sr.notify_developer:
            log.error("NEEDS ATTENTION: sending failed [%s] %s", sr.error_code, sr.detail)
        return sr

    def is_done(sr) -> bool:  # True = finished, False = leave unread and retry on the next poll
        return sr.ok or sr.error_code in PERMANENT_SEND_ERRORS

    try:
        # 1. Understand the request
        parsed = parse_request(em.body, em.subject)
        if not parsed.ok:
            if parsed.notify_developer:
                log.error("NEEDS ATTENTION: parser [%s] %s", parsed.error_code, parsed.detail)
            return is_done(send(build_plain_body(parsed.user_message, name)))
        log.info("Request from %s: %s / %s", em.from_addr, parsed.matter_number, parsed.document_type)

        # 2. Fetch the documents
        result = fetch_documents(parsed.matter_number, parsed.document_type)
        try:
            if not result.ok:
                if result.notify_developer:
                    log.error("NEEDS ATTENTION: scraper [%s] %s", result.error_code, result.detail)
                return is_done(send(build_plain_body(result.user_message, name)))

            # 3. Metadata, then ZIP + reply (with fallbacks if a big attachment can't be sent)
            metadata = extract_metadata(result.header_text)
            return is_done(deliver(result, metadata, name, send))
        finally:
            if result.work_dir:  # always delete the downloaded files
                shutil.rmtree(result.work_dir, ignore_errors=True)

    except Exception:
        log.exception("Unexpected error while handling %s", em.key)
        if replied["ok"]:
            return True  # the user already got their answer
        return is_done(send(build_plain_body(GENERIC_ERROR, name)))


def process_one(em, store: ProcessedStore, limiter: RateLimiter, attempts: dict, allowed: set):
    def finish():
        store.add(em.key)  # remember first, so a failed mark_seen can't cause a duplicate reply
        try:
            mark_seen(em.uid)
        except (imaplib.IMAP4.error, OSError, InboxError) as exc:
            log.warning("Couldn't mark uid=%s as read (%r); it's remembered, so it won't be answered twice",
                        em.uid, exc)

    if em.key in store:
        finish()
        return
    if em.skip_reason:
        log.info("Ignoring email from %r: %s", em.from_addr, em.skip_reason)
        finish()
        return
    if allowed and em.from_addr.lower() not in allowed:
        log.info("Ignoring email from %r: not in ALLOWED_SENDERS", em.from_addr)
        finish()
        return
    if attempts[em.key] == 0 and not limiter.allow(em.from_addr.lower()):
        log.warning("Ignoring email from %r: over %d requests/hour", em.from_addr,
                    MAX_REQUESTS_PER_SENDER_PER_HOUR)
        finish()
        return

    attempts[em.key] += 1
    log.info("Handling email from %r, subject %r (attempt %d/%d)", em.from_addr,
             em.subject[:60], attempts[em.key], MAX_ATTEMPTS)
    try:
        done = handle_message(em)
    except Exception:  # handle_message has its own safety net; this is the last resort
        log.exception("handle_message crashed for %s", em.key)
        done = False

    if done:
        finish()
    elif attempts[em.key] >= MAX_ATTEMPTS:
        log.error("NEEDS ATTENTION: giving up on email from %r after %d attempts", em.from_addr, MAX_ATTEMPTS)
        finish()
    else:
        log.warning("Reply could not be sent; will retry on the next poll")


# ================================================================ the loop
def run(once: bool = False, sleep=time.sleep):
    store = ProcessedStore(PROCESSED_FILE)
    limiter = RateLimiter(MAX_REQUESTS_PER_SENDER_PER_HOUR)
    attempts = defaultdict(int)
    allowed = {a.strip().lower() for a in os.getenv("ALLOWED_SENDERS", "").split(",") if a.strip()}
    inbox_failures = 0
    log.info("Agent running. Checking the inbox every %ds%s", POLL_INTERVAL_S,
             f" (only answering: {', '.join(sorted(allowed))})" if allowed else "")

    while True:
        try:
            emails = fetch_unread()
            inbox_failures = 0
        except (imaplib.IMAP4.error, OSError, InboxError) as exc:
            inbox_failures += 1
            delay = min(300, POLL_INTERVAL_S * 2 ** min(inbox_failures, 4))  # back off, cap at 5 min
            log.error("Couldn't read the inbox (%r). Retrying in %ds", exc, delay)
            if once:
                return
            sleep(delay)
            continue

        if emails:
            log.info("%d unread email(s)", len(emails))
        for em in emails:
            process_one(em, store, limiter, attempts, allowed)
        if once:
            return
        sleep(POLL_INTERVAL_S)


def main():
    setup_logging()
    check_config()
    try:
        check_login()
    except (imaplib.IMAP4.error, OSError, InboxError) as exc:
        sys.exit(f"Couldn't log in to Gmail over IMAP: {exc!r}\n"
                 "Check GMAIL_ADDRESS / GMAIL_APP_PASSWORD, and that IMAP is enabled in Gmail settings "
                 "(Settings > See all settings > Forwarding and POP/IMAP).")
    keep_awake()
    try:
        run(once="--once" in sys.argv)
    except KeyboardInterrupt:
        log.info("Stopped by user")


if __name__ == "__main__":
    main()