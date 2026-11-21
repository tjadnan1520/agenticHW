"""Clean model output and accept only one plain SELECT or INSERT (first safety layer)."""

import re


class SqlRejected(Exception):
    """The statement is not allowed. Retrying with the model will not help."""

    retryable = False


class FixableSql(SqlRejected):
    """Allowed kind of statement, but with a mistake the model can correct."""

    retryable = True


# Strings, quoted identifiers and comments are masked in one pass, so a word
# inside 'lunch; drop' is not mistaken for code.
_MASKED = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|--[^\n]*|/\*.*?\*/", re.S)
_FENCE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.S)
_START = re.compile(r"\b(WITH|SELECT|INSERT)\b", re.I)
_FORBIDDEN = re.compile(
    r"\b(DROP|DELETE|UPDATE|ALTER|CREATE|PRAGMA|ATTACH|DETACH|VACUUM|REINDEX|"
    r"TRIGGER|LOAD_EXTENSION|SAVEPOINT|RELEASE|ROLLBACK|BEGIN|COMMIT|sqlite_\w+)\b"
    r"|\bOR\s+REPLACE\b|\bREPLACE\s+INTO\b",
    re.I,
)


# Small models invent modifiers such as 'localmonth'; SQLite quietly returns NULL for them.
_DATE_CALL = re.compile(r"\b(?:date|datetime|time)\s*\(([^)]*)\)", re.I)
_QUOTED = re.compile(r"'([^']*)'")
_GOOD_ARG = re.compile(
    r"now|\d{4}-\d\d-\d\d(?: \d\d:\d\d(?::\d\d)?)?|localtime|utc|start of (?:day|month|year)|"
    r"weekday [0-6]|[+-]?\d+ (?:second|minute|hour|day|month|year)s?",
    re.I,
)


def _check_date_modifiers(sql: str) -> None:
    for call in _DATE_CALL.finditer(sql):
        for arg in _QUOTED.findall(call.group(1)):
            if not _GOOD_ARG.fullmatch(arg.strip()):
                raise FixableSql(
                    f"invalid date modifier '{arg}'. Use only: 'now', 'localtime', "
                    "'start of month', 'start of year', '-N day', '-N month'"
                )


def _mask(sql: str) -> str:
    return _MASKED.sub(lambda m: "x" * len(m.group(0)), sql)


def extract_sql(raw: str) -> str:
    """Strip markdown fences, chatter before the statement, and anything after the first ';'."""
    text = raw.strip()
    fenced = _FENCE.search(text)
    text = fenced.group(1).strip() if fenced else text.replace("```", "")
    start = _START.search(text)
    if start:
        text = text[start.start():]
    elif "UNSUPPORTED" in text.upper():
        return "SELECT 'UNSUPPORTED'"  # the model forgot the SELECT part of the sentinel
    end = _mask(text).find(";")
    return (text[:end] if end != -1 else text).strip()


def validate(sql: str) -> str:
    """Return 'SELECT' or 'INSERT', or raise SqlRejected."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise SqlRejected("empty statement")
    masked = _mask(cleaned)
    if ";" in masked:
        raise SqlRejected("only one statement is allowed")
    if "'" in masked or '"' in masked or "/*" in masked:
        raise SqlRejected("unbalanced quote or comment")
    first = masked.split(None, 1)[0].upper()
    if first in ("SELECT", "WITH"):
        kind = "SELECT"
    elif first == "INSERT":
        kind = "INSERT"
    else:
        raise SqlRejected(f"only SELECT and INSERT are allowed, got {first}")
    if kind == "SELECT" and re.search(r"\bINSERT\b", masked, re.I):
        raise SqlRejected("INSERT inside a SELECT is not allowed")
    bad = _FORBIDDEN.search(masked)
    if bad:
        raise SqlRejected(f"forbidden keyword: {bad.group(0).strip()}")
    _check_date_modifiers(cleaned)
    return kind
