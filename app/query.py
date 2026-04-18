"""Policy refusals + LLM-driven query router."""
from __future__ import annotations
import json
from . import mistral


_POLICY_SYS = """You are a sensitive-data filter for a document Q&A system. Users upload documents (CVs, contracts, reports) that may accidentally contain highly sensitive personal identifiers the uploader did not intend to surface. Your only job is to refuse queries that try to EXTRACT that kind of identifier from the documents.

BLOCK when the user's message is asking the system to reveal, list, or look up any of the following from the documents:
- Social Security Numbers / national ID numbers
- Bank account numbers, IBANs, routing numbers
- Credit or debit card numbers, CVVs, PINs
- Passwords, API keys, access tokens, private cryptographic keys
- Passport numbers, driver's license numbers

DO NOT block:
- Email addresses, phone numbers, mailing addresses, websites, social media handles (these are routine contact info and are fine to surface)
- Names, titles, affiliations, dates of birth alone, job histories, education
- Questions about document topic, content, methodology, results, opinions

OUTPUT JSON ONLY:
{"blocked": true | false, "reason": string}

If blocked, "reason" is one short sentence shown to the user.
If not blocked, "reason" may be empty.
"""


def screen_for_sensitive_data(q: str) -> str | None:
    try:
        raw = mistral.chat(
            [{"role": "system", "content": _POLICY_SYS},
             {"role": "user", "content": q}],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("blocked"):
        return None
    reason = str(data.get("reason") or "").strip()
    return reason or "I can't help with that request."


_ROUTER_SYS = """You are a query router for a document Q&A system. The user has uploaded PDFs into a knowledge base. Classify their message into EXACTLY ONE route and output JSON.

ROUTES:

1. "chat" — the message is one of:
   - a greeting, farewell, or acknowledgement (hi, thanks, bye)
   - a general-knowledge or factual question that does NOT require the user's documents (e.g. "capital of France", "who wrote Hamlet", "2+2")
   - a request for your opinion
   - a meta-question about what this tool does
   Produce a "response" field: a direct, plain answer in 1-2 sentences.
   STRICT: the response must NOT mention documents, PDFs, uploads, knowledge base, or the purpose of this system UNLESS the user explicitly asked what the tool does. Just answer the question. If asked "what is the capital of France?", the response is literally "Paris." — nothing more.

2. "kb" — the message is a specific, factual question likely answerable from the uploaded documents.
   Produce two fields:
   - "rewrite": a concise, keyword-rich search query. Expand acronyms, resolve pronouns, keep proper nouns, and add synonyms that might appear verbatim in the source text (e.g. for a question about authorship, include words like "authors", "affiliation", "institution"). Drop filler words.
   - "format": "qa" for a prose answer, "list" when the user asks for bullets or enumeration, "table" for comparisons or tabular output.

3. "clarify" — the message is too vague, under-specified, or ambiguous to search for usefully (e.g. "tell me more", "summarize it" with no obvious referent, pronouns with no antecedent).
   Produce a "question" field: a short clarifying question.

DECISION GUIDE:
- If you can extract at least one concrete entity, topic, or proper noun to search for, prefer "kb" over "clarify".
- If the message could be answered either from general knowledge or the docs, prefer "kb" (the system will refuse gracefully if nothing relevant is found).

OUTPUT: JSON only, no preamble. Schema:
{"route": "chat" | "kb" | "clarify",
 "response": string,   // required when route is "chat"
 "question": string,   // required when route is "clarify"
 "rewrite":  string,   // required when route is "kb"
 "format":   "qa" | "list" | "table"}   // required when route is "kb"
"""


def route_query(q: str) -> dict:
    messages = [
        {"role": "system", "content": _ROUTER_SYS},
        {"role": "user", "content": q},
    ]
    try:
        raw = mistral.chat(
            messages,
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        data = json.loads(raw)
    except Exception:
        return {"route": "kb", "rewrite": q, "format": "qa"}
    if not isinstance(data, dict) or data.get("route") not in {"chat", "kb", "clarify"}:
        return {"route": "kb", "rewrite": q, "format": "qa"}
    return data
