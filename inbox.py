"""
inbox.py - Read unread emails from the agent's Gmail inbox over IMAP.

    emails = fetch_unread()   # list[InboundEmail]. Nothing is marked as read yet.
    mark_seen(email.uid)      # call this once you've dealt with the email

Every function opens a short connection and closes it again. That way a long scrape between
calls can't leave an idle connection to time out. Messages are fetched with BODY.PEEK, which
does NOT mark them as read, so if the agent crashes mid-request the email is picked up again.

Environment variables (.env): GMAIL_ADDRESS, GMAIL_APP_PASSWORD (same ones mailer.py uses)
Gmail setup: IMAP must be enabled (Settings > See all settings > Forwarding and POP/IMAP).
"""
import imaplib
import logging
import os
import re
from dataclasses import dataclass
from email import message_from_bytes, policy
from email.utils import parseaddr
from html.parser import HTMLParser
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
IMAP_TIMEOUT_S = 30
MAX_PER_POLL = 20  # oldest first; anything beyond this waits for the next poll

# Senders that are never a real person asking for documents.
_AUTOMATED_LOCAL_PARTS = ("no-reply", "noreply", "donotreply", "do-not-reply",
                          "mailer-daemon", "postmaster", "bounce")
_AUTOMATED_HEADERS = ("X-Autoreply", "X-Autorespond", "X-Failed-Recipients", "List-Id", "List-Unsubscribe")
# Deliberately NOT here: X-Auto-Response-Suppress. Outlook/Exchange adds it to ordinary mail.


class InboxError(Exception):
    pass


@dataclass
class InboundEmail:
    uid: str  # IMAP UID, used to mark the message as read
    key: str  # stable de-duplication key (Message-ID, or the UID if there isn't one)
    message_id: Optional[str]  # for threading the reply
    references: Optional[str]
    from_header: str  # e.g. 'Jane Doe <jane@example.com>'
    from_addr: str  # e.g. 'jane@example.com'
    subject: str
    body: str  # plain text (HTML-only emails are converted)
    skip_reason: Optional[str] = None  # set if this should NOT get a reply (automated, self, ...)


# ---------------------------------------------------------------- IMAP plumbing
def _credentials():
    user = os.getenv("GMAIL_ADDRESS", "").strip()
    password = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "").strip()
    if not user or not password:
        raise InboxError("GMAIL_ADDRESS or GMAIL_APP_PASSWORD is not set")
    return user, password


def _close(imap):
    try:
        imap.logout()
    except Exception:
        pass


def _connect():
    user, password = _credentials()
    imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT, timeout=IMAP_TIMEOUT_S)
    try:
        imap.login(user, password)
        status, _ = imap.select("INBOX")  # read-write, so we can set the Seen flag later
        if status != "OK":
            raise InboxError("couldn't open the INBOX")
    except Exception:
        _close(imap)
        raise
    return imap


def check_login():
    """Raises if we can't log in. Call once at startup so problems show up immediately."""
    _close(_connect())


# ---------------------------------------------------------------- parsing
class _HTMLText(HTMLParser):
    _BLOCKS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _html_to_text(html: str) -> str:
    parser = _HTMLText()
    parser.feed(html)
    parser.close()
    text = re.sub(r"[ \t]+", " ", "".join(parser.parts))
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _body_text(msg) -> str:
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    try:
        content = part.get_content()
    except (LookupError, UnicodeDecodeError, ValueError):  # unknown charset etc.
        payload = part.get_payload(decode=True) or b""
        content = payload.decode("utf-8", errors="replace")
    if part.get_content_type() == "text/html":
        content = _html_to_text(content)
    return content.strip()


def _skip_reason(msg, from_addr: str, own_address: str) -> Optional[str]:
    """Return why we should NOT reply to this message, or None if it looks like a real person."""
    if not from_addr:
        return "no sender address"
    if from_addr.lower() == own_address.lower():
        return "sent by the agent itself"
    local = from_addr.split("@")[0].lower()
    if local.startswith(_AUTOMATED_LOCAL_PARTS):
        return f"automated sender ({local})"
    auto = str(msg.get("Auto-Submitted") or "no").strip().lower()
    if auto != "no":
        return f"Auto-Submitted: {auto}"
    if str(msg.get("Precedence") or "").strip().lower() in {"bulk", "junk", "list", "auto_reply"}:
        return "bulk/auto-reply Precedence header"
    for header in _AUTOMATED_HEADERS:
        if msg.get(header):
            return f"{header} header present"
    if msg.get_content_type() == "multipart/report":
        return "delivery status / bounce report"
    return None


def _parse_email(uid: str, raw: bytes, own_address: str) -> InboundEmail:
    try:
        msg = message_from_bytes(raw, policy=policy.default)
        from_header = str(msg.get("From") or "").strip()
        _, from_addr = parseaddr(from_header)
        message_id = str(msg.get("Message-ID") or "").strip() or None
        return InboundEmail(
            uid=uid,
            key=message_id or f"uid:{uid}",
            message_id=message_id,
            references=str(msg.get("References") or "").strip() or None,
            from_header=from_header,
            from_addr=from_addr.strip(),
            subject=str(msg.get("Subject") or "").strip(),
            body=_body_text(msg),
            skip_reason=_skip_reason(msg, from_addr.strip(), own_address),
        )
    except Exception as exc:  # a malformed email must never crash the agent
        log.warning("Couldn't parse email uid=%s: %r", uid, exc)
        return InboundEmail(uid=uid, key=f"uid:{uid}", message_id=None, references=None,
                            from_header="", from_addr="", subject="", body="",
                            skip_reason="could not be parsed")


# ---------------------------------------------------------------- public API
def fetch_unread(limit: int = MAX_PER_POLL) -> list:
    """Unread emails in the INBOX, oldest first. Raises imaplib.IMAP4.error / OSError / InboxError
    if the mailbox can't be reached; the caller should catch those and retry later."""
    own_address, _ = _credentials()
    imap = _connect()
    try:
        status, data = imap.uid("SEARCH", None, "UNSEEN")
        if status != "OK":
            raise InboxError(f"UNSEEN search failed: {status}")
        emails = []
        for uid in data[0].split()[:limit]:
            uid = uid.decode()
            status, parts = imap.uid("FETCH", uid, "(BODY.PEEK[])")  # PEEK = don't mark as read
            if status != "OK" or not parts or not isinstance(parts[0], tuple):
                log.warning("Couldn't fetch uid=%s (%s)", uid, status)
                continue
            emails.append(_parse_email(uid, parts[0][1], own_address))
        return emails
    finally:
        _close(imap)


def mark_seen(uid: str):
    """Mark one message as read. Raises on connection problems."""
    imap = _connect()
    try:
        status, _ = imap.uid("STORE", uid, "+FLAGS", "(\\Seen)")
        if status != "OK":
            raise InboxError(f"couldn't mark uid={uid} as read")
    finally:
        _close(imap)