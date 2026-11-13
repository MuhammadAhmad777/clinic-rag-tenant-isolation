import html
import re

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) (instructions|rules)",
    r"system override",
    r"you are now",
    r"admin mode",
    r"disregard (the )?(system|previous)",
    r"reveal .*(tenant|clinic|secret|password|key)",
    r"access to (every|all) tenant",
    r"</?(system|document|instructions)>",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
SOURCE_ID_RE = re.compile(r"\b(?:KB|TCK|LOG)-[A-Z]{3}-\d{3,6}\b")


def flag_injection(text):
    return sorted({m.group(0).lower() for m in _INJECTION_RE.finditer(text)})


def wrap_document(record_id, status, title, text):
    # escape markup so a document can't close its own tag and pose as instructions
    return (f'<document id="{html.escape(record_id)}" status="{html.escape(status)}">\n'
            f"{html.escape(title)}: {html.escape(text)}\n</document>")


def guard_output(answer, allowed_ids, foreign_terms):
    flags = []

    def check_id(m):
        if m.group(0) in allowed_ids:
            return m.group(0)
        flags.append(f"blocked_unretrieved_citation:{m.group(0)}")
        return "[REDACTED-SOURCE]"

    cleaned = SOURCE_ID_RE.sub(check_id, answer)
    for term in foreign_terms:
        if term and re.search(re.escape(term), cleaned, re.IGNORECASE):
            cleaned = re.sub(re.escape(term), "[REDACTED-TENANT]", cleaned, flags=re.IGNORECASE)
            flags.append(f"blocked_cross_tenant_term:{term}")
    return cleaned, flags


def cited_ids(answer):
    seen = []
    for m in SOURCE_ID_RE.finditer(answer):
        if m.group(0) not in seen:
            seen.append(m.group(0))
    return seen