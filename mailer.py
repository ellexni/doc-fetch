"""
mailer.py - ZIP the downloaded files and send emails through Gmail SMTP.

    zr = make_zip(result.files, result.work_dir, "M12205", "Other Documents")
    sr = send_email(to_addr, subject, body, attachment_path=zr.zip_path, in_reply_to=message_id)

Both functions NEVER raise. Always check .ok on what they return.

Environment variables (.env):
  GMAIL_ADDRESS        the agent's Gmail address
  GMAIL_APP_PASSWORD   the 16-character Google app password (spaces are fine, they get stripped)
  SMTP_PORT            optional. 465 (default, SSL) or 587 (STARTTLS). Try 587 if big emails keep
                       failing on your network.
"""
import logging
import os
import re
import smtplib
import time
import zipfile
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))  # 465 = SSL, 587 = STARTTLS
# The whole email is uploaded in one go, so the timeout grows with the attachment size.
SMTP_BASE_TIMEOUT_S = 60
SMTP_TIMEOUT_PER_MB_S = 15
SMTP_MAX_TIMEOUT_S = 600

# Gmail's limit is about 25 MB per message, and email encoding (base64) makes attachments
# roughly a third bigger, so keep the ZIP itself comfortably below ~18 MB.
MAX_ZIP_BYTES = 18 * 1024 * 1024
ZIP_OVERHEAD_PER_FILE = 512  # bytes reserved per file for ZIP headers


# ================================================================ ZIP
@dataclass
class ZipResult:
    ok: bool
    zip_path: Optional[Path] = None
    included: list = field(default_factory=list)  # Paths that went into the ZIP
    skipped_too_large: list = field(default_factory=list)  # [(filename, size_in_bytes)]
    size_bytes: int = 0
    error_code: Optional[str] = None  # NO_FILES | ZIP_TOO_LARGE | ZIP_ERROR
    detail: Optional[str] = None  # logs only


def _write_zip(files, out_dir: Path, zip_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    return zip_path


def make_zip(files, out_dir, matter: str, doc_type: str, max_bytes: int = MAX_ZIP_BYTES) -> ZipResult:
    """ZIP as many of `files` as fit under max_bytes, keeping their original order.

    A file that would push the ZIP over the limit is skipped (later, smaller files can still
    fit), and is reported in skipped_too_large so the email can mention it."""
    try:
        files = [Path(f) for f in (files or []) if Path(f).is_file()]
        if not files:
            return ZipResult(ok=False, error_code="NO_FILES", detail="no files to zip")

        # Use uncompressed sizes: a safe upper bound on what the ZIP will weigh.
        chosen, skipped, budget = [], [], 0
        for f in files:
            size = f.stat().st_size
            cost = size + ZIP_OVERHEAD_PER_FILE
            if budget + cost <= max_bytes:
                chosen.append(f)
                budget += cost
            else:
                skipped.append((f.name, size))

        slug = re.sub(r"[^A-Za-z0-9]+", "_", doc_type).strip("_")
        zip_name = f"{matter}_{slug}.zip"
        out_dir = Path(out_dir)
        while chosen:
            zip_path = _write_zip(chosen, out_dir, zip_name)
            actual = zip_path.stat().st_size
            if actual <= max_bytes:
                log.info("ZIP ready: %s (%d files, %.1f MB, %d left out)", zip_name, len(chosen),
                         actual / 1e6, len(skipped))
                return ZipResult(ok=True, zip_path=zip_path, included=chosen,
                                 skipped_too_large=skipped, size_bytes=actual)
            # Shouldn't happen (the estimate is conservative), but drop the last file and retry.
            dropped = chosen.pop()
            skipped.append((dropped.name, dropped.stat().st_size))
            zip_path.unlink(missing_ok=True)

        return ZipResult(ok=False, skipped_too_large=skipped, error_code="ZIP_TOO_LARGE",
                         detail=f"every file exceeds the {max_bytes} byte limit: {skipped}")
    except Exception as exc:  # disk full, permissions, corrupt file...
        log.exception("Unexpected error while zipping")
        return ZipResult(ok=False, error_code="ZIP_ERROR", detail=repr(exc))


# ================================================================ SEND
@dataclass
class SendResult:
    ok: bool
    error_code: Optional[str] = None
    detail: Optional[str] = None  # logs only
    notify_developer: bool = False  # True for problems you should look at (bad password, etc.)


def _send_fail(code, detail, dev=False) -> SendResult:
    log.warning("Send failed [%s] %s", code, detail)
    return SendResult(ok=False, error_code=code, detail=detail, notify_developer=dev)


def _one_line(text: str, limit: int = 250) -> str:
    """Strip line breaks so user-controlled text can't inject extra email headers."""
    return re.sub(r"[\r\n]+", " ", text or "").strip()[:limit]


def _open_smtp(timeout: int):
    """Connect to Gmail: SSL on port 465 (default), or STARTTLS if SMTP_PORT=587."""
    if SMTP_PORT == 465:
        return smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=timeout)
    smtp = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=timeout)
    try:
        smtp.starttls()
    except Exception:
        smtp.close()
        raise
    return smtp


def send_email(to_addr: str, subject: str, body: str, attachment_path=None,
               in_reply_to: Optional[str] = None, references: Optional[str] = None) -> SendResult:
    """Send a plain-text email, optionally with a ZIP attached. Never raises.

    Pass in_reply_to (the original email's Message-ID header) to keep the reply in the
    same conversation thread."""
    try:
        user = os.getenv("GMAIL_ADDRESS", "").strip()
        password = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "").strip()
        if not user or not password:
            return _send_fail("SEND_CONFIG_ERROR", "GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set", True)

        _, addr = parseaddr(_one_line(to_addr))
        if not re.fullmatch(r"[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+", addr or ""):
            return _send_fail("INVALID_RECIPIENT", f"bad recipient address: {to_addr!r}")

        msg = EmailMessage()
        msg["From"] = user
        msg["To"] = addr
        msg["Subject"] = _one_line(subject, 200) or "Your document request"
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid()
        if in_reply_to:
            msg["In-Reply-To"] = _one_line(in_reply_to)
            msg["References"] = _one_line(references or in_reply_to, 1000)
        msg.set_content(body or "")

        size_mb = 0.0
        if attachment_path:
            path = Path(attachment_path)
            try:
                data = path.read_bytes()
            except OSError as exc:
                return _send_fail("ATTACHMENT_ERROR", f"can't read {path}: {exc}", True)
            if len(data) > MAX_ZIP_BYTES:
                return _send_fail("ATTACHMENT_TOO_LARGE", f"{path.name} is {len(data)} bytes")
            msg.add_attachment(data, maintype="application", subtype="zip", filename=path.name)
            size_mb = len(data) / 1e6
        timeout = int(min(SMTP_MAX_TIMEOUT_S, SMTP_BASE_TIMEOUT_S + SMTP_TIMEOUT_PER_MB_S * size_mb))

        for attempt in (1, 2):
            started = time.monotonic()
            try:
                with _open_smtp(timeout) as smtp:
                    smtp.login(user, password)
                    smtp.send_message(msg)
                log.info("Email sent to %s (attachment: %s)", addr,
                         Path(attachment_path).name if attachment_path else "none")
                return SendResult(ok=True)
            except smtplib.SMTPAuthenticationError as exc:  # wrong/revoked app password
                return _send_fail("SEND_AUTH_ERROR", repr(exc), True)
            except smtplib.SMTPRecipientsRefused as exc:  # address doesn't exist, etc.
                return _send_fail("RECIPIENT_REFUSED", repr(exc))
            except (smtplib.SMTPSenderRefused, smtplib.SMTPDataError) as exc:  # too big, blocked
                return _send_fail("MESSAGE_REJECTED", repr(exc), True)
            except (smtplib.SMTPException, OSError) as exc:  # timeouts, dropped connections
                elapsed = time.monotonic() - started
                log.warning("Send attempt %d failed after %.0fs (attachment %.1f MB, timeout %ds): %r",
                            attempt, elapsed, size_mb, timeout, exc)
                timed_out = elapsed >= timeout * 0.9  # retrying a too-slow upload just wastes minutes
                if attempt == 1 and not timed_out:
                    time.sleep(3)
                    continue
                return _send_fail("SEND_FAILED",
                                  f"{exc!r} (after {elapsed:.0f}s, timeout {timeout}s, {size_mb:.1f} MB)", True)
        return _send_fail("SEND_FAILED", "exhausted attempts", True)  # not reachable; safety net
    except Exception as exc:  # e.g. ValueError from an invalid header
        log.exception("Unexpected error while sending email")
        return _send_fail("UNEXPECTED_ERROR", repr(exc), True)