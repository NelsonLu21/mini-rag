from __future__ import annotations
import os, time, httpx
from typing import List
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("MISTRAL_API_KEY")
if not API_KEY:
    raise RuntimeError("MISTRAL_API_KEY is not set. Copy .env.example to .env and add your key.")

BASE = "https://api.mistral.ai/v1"
EMBED_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-small-latest"
OCR_MODEL = "mistral-ocr-latest"

_client = httpx.Client(timeout=60.0, headers={"Authorization": f"Bearer {API_KEY}"})

_RETRY_STATUSES = {429, 500, 502, 503, 504}


def _post_retry(url: str, *, retries: int = 3, **kwargs) -> httpx.Response:
    last: httpx.Response | None = None
    for attempt in range(retries + 1):
        try:
            r = _client.post(url, **kwargs)
        except httpx.RequestError:
            if attempt == retries:
                raise
            time.sleep(0.5 * (2 ** attempt))
            continue
        if r.status_code not in _RETRY_STATUSES or attempt == retries:
            return r
        last = r
        time.sleep(0.5 * (2 ** attempt))
    assert last is not None
    return last


def embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    out: List[List[float]] = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        r = _post_retry(f"{BASE}/embeddings", json={"model": EMBED_MODEL, "input": batch})
        r.raise_for_status()
        out.extend(d["embedding"] for d in r.json()["data"])
    return out


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 800, response_format: dict | None = None) -> str:
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    r = _post_retry(f"{BASE}/chat/completions", json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ocr_file(path: str, filename: str) -> list[dict]:
    with open(path, "rb") as fh:
        up = _post_retry(
            f"{BASE}/files",
            files={"file": (filename, fh, "application/pdf")},
            data={"purpose": "ocr"},
            timeout=120.0,
        )
    up.raise_for_status()
    file_id = up.json()["id"]

    url_r = _client.get(f"{BASE}/files/{file_id}/url", params={"expiry": 24})
    url_r.raise_for_status()
    signed = url_r.json()["url"]

    r = _post_retry(
        f"{BASE}/ocr",
        json={
            "model": OCR_MODEL,
            "document": {"type": "document_url", "document_url": signed},
        },
        timeout=300.0,
    )
    r.raise_for_status()
    return r.json().get("pages", [])
