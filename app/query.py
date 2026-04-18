"""Intent detection, query rewriting, refusal policies."""
from __future__ import annotations
import re
from . import mistral

GREET = {"hi", "hello", "hey", "yo", "thanks", "thank you", "bye", "goodbye", "ok", "okay"}

PII_RX = re.compile(
    r"\b(\d{3}-\d{2}-\d{4}|\d{16}|\d{4}\s?\d{4}\s?\d{4}\s?\d{4})\b"  # SSN / card
    r"|\b[\w.+-]+@[\w-]+\.[\w.-]+\b"  # email
)
LEGAL_RX = re.compile(r"\b(sue|lawsuit|legal advice|liabilit|attorney)\b", re.I)
MEDICAL_RX = re.compile(r"\b(diagnose|prescribe|dosage|symptom|treatment plan)\b", re.I)


def classify_intent(q: str) -> str:
    """Return one of: greeting, list, table, qa."""
    s = q.strip().lower()
    if not s or s.rstrip("!.?") in GREET or len(s.split()) <= 2 and any(g in s for g in GREET):
        return "greeting"
    if re.search(r"\b(list|enumerate|bullet)\b", s):
        return "list"
    if re.search(r"\b(table|compare|columns?)\b", s):
        return "table"
    return "qa"


def policy_check(q: str) -> str | None:
    """Return a refusal message if the query violates policy, else None."""
    if PII_RX.search(q):
        return "I can't process queries that contain personal identifiers (e.g. SSN, credit card, email). Please remove them and try again."
    if LEGAL_RX.search(q):
        return "I can share what the documents say, but I can't provide legal advice. For legal matters, consult a qualified attorney."
    if MEDICAL_RX.search(q):
        return "I can share what the documents say, but I can't provide medical advice. For health decisions, consult a licensed clinician."
    return None


def transform(q: str) -> str:
    """Rewrite the user query into a self-contained search query.

    Falls back to the original on API failure — retrieval still works.
    """
    try:
        out = mistral.chat(
            [
                {"role": "system", "content": (
                    "Rewrite the user's question into a concise, keyword-rich search "
                    "query for a document retrieval system. Expand acronyms, resolve "
                    "pronouns, keep proper nouns. Output the query only — no preamble."
                )},
                {"role": "user", "content": q},
            ],
            temperature=0.0,
            max_tokens=80,
        ).strip().strip('"')
        return out or q
    except Exception:
        return q
