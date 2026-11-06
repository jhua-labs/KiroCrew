"""The tokens the batch write guard compares: values a write quotes that only a
call's result or an earlier message could supply.

A token is a sha or another hex id, a Kiro Crew item, session or artifact id, a
long run, review or job id, or a PR number written as one (a hash and digits,
or a ``pull/`` path). A clock time is not a token, since a model stamps its own
reports with one; a time to the second counts only as proof (:func:`times_of`).
Kept apart from :mod:`kiro_crew.acp.batch_guard`, whose glob matcher must stay free
of regular expressions.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["MAX_SCANNED_TEXT_CHARS", "is_minted", "times_of", "token_keys", "tokens_of"]

#: Characters of one text :func:`tokens_of` reads. A token past the bound is not
#: seen, which can only make a later write citing it look unseen; a result that
#: long is past what a harness shows the model anyway.
MAX_SCANNED_TEXT_CHARS = 256 * 1024

_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    # A sha, a sha256 digest or another hex id: 7-64 characters with a digit and
    # a letter, in either case (read lowercase).
    re.compile(r"\b((?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,64})\b", re.IGNORECASE),
    re.compile(r"\b(it_[0-9a-f]{8})\b"),
    re.compile(r"\b(chat-\d{1,12}-\d{1,12})\b"),
    re.compile(r"\b(art_[0-9a-f]{8,32})\b"),
    # Run, review, job and event ids.
    re.compile(r"(?<![\w-])(\d{10,20})\b"),
    # PR and issue numbers where they are named as such.
    re.compile(r"#(\d{2,6})\b"),
    re.compile(r"\bpull/(\d{1,7})\b"),
    re.compile(r"\bruns/(\d{1,20})\b"),
    re.compile(r"\"pr\": ?(\d{1,7})\b"),
    re.compile(r"\bgh pr (?:merge|edit|comment|ready|close|review|view|checks) (\d{1,7})\b"),
)
_HEX = re.compile(r"[0-9a-f]{7,64}")
#: The shortest prefix a hex token is also known by.
_MIN_HEX_PREFIX = 7
#: The longest sha prefix stored for a seen sha (shortcut: a 13-39 character
#: cut is not matched; widen this if writes quote such cuts).
_MAX_HEX_PREFIX = 12


def _plain(text: str) -> str:
    """*text*, bounded, with JSON string escapes for line breaks and tabs read as
    blanks and escaped quotes read as quotes: a JSON-encoded result keeps
    ``abc1234\\ndef5678`` as one run of word characters, which would hide the
    second id, and turns ``"pr": 1742`` into ``\\"pr\\": 1742``."""
    text = text[:MAX_SCANNED_TEXT_CHARS]
    if "\\" in text:
        text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
        text = text.replace('\\"', '"')
    return text


def tokens_of(text: Any) -> set[str]:
    """The tokens in *text* (its first :data:`MAX_SCANNED_TEXT_CHARS`)."""
    if not isinstance(text, str) or not text:
        return set()
    text = _plain(text)
    out = {m.group(1) for pattern in _TOKEN_PATTERNS[1:] for m in pattern.finditer(text)}
    out.update(m.group(1).lower() for m in _TOKEN_PATTERNS[0].finditer(text))
    return out


#: A clock time to the second (``07:30:45Z``, ``07:30:45.120Z``). Only ever proof:
#: a write citing one is refused only when a returned call of its batch printed it.
_TIME = re.compile(r"(?<![\d:])(\d{2}:\d{2}:\d{2})(?:\.\d+)?Z?(?![\d:])")


def times_of(text: Any) -> set[str]:
    """The to-the-second clock times in *text* (its first :data:`MAX_SCANNED_TEXT_CHARS`)."""
    if not isinstance(text, str) or not text:
        return set()
    return set(_TIME.findall(_plain(text)))


def is_hex(token: str) -> bool:
    """Whether *token* is a hex id (a sha or a cut of one)."""
    return _HEX.fullmatch(token) is not None


def token_keys(token: str) -> list[str]:
    """The keys a SEEN *token* is stored under: itself and, for a hex token, its
    prefixes of 7 to 12 characters (the lengths git abbreviates a sha to), so a
    write that shortens a seen sha that way matches. Seven keys per sha at most,
    so one result of :data:`MAX_BATCH_TOKENS` shas fits the session's seen set.

    Matching runs one way. A quoted token is looked up as itself, never by its own
    prefixes, so a seen short prefix never makes a longer quoted sha seen."""
    if _HEX.fullmatch(token):
        end = min(len(token), _MAX_HEX_PREFIX + 1)
        return [token] + [token[:n] for n in range(_MIN_HEX_PREFIX, end)]
    return [token]


#: The id families Kiro Crew mints: a session key and a ledger item id.
_MINTED = re.compile(r"chat-\d{1,12}-\d{1,12}|it_[0-9a-f]{8}")


def is_minted(token: str) -> bool:
    """Whether *token* is a session key or a ledger item id. A resumed session
    still refuses one its resumed text never showed: unlike a sha, such an id is
    new when a same-batch create returns it."""
    return bool(_MINTED.fullmatch(token))
