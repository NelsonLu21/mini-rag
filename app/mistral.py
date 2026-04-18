"""Thin Mistral API client. No SDK dependency — just httpx."""
from __future__ import annotations
import os, httpx
from typing import List
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("MISTRAL_API_KEY")
if not API_KEY:
    raise RuntimeError("MISTRAL_API_KEY is not set. Copy .env.example to .env and add your key.")

BASE = "https://api.mistral.ai/v1"
EMBED_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-small-latest"

_client = httpx.Client(timeout=60.0, headers={"Authorization": f"Bearer {API_KEY}"})


def embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    out: List[List[float]] = []
    # batch of 32 to stay well under request limits
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        r = _client.post(f"{BASE}/embeddings", json={"model": EMBED_MODEL, "input": batch})
        r.raise_for_status()
        out.extend(d["embedding"] for d in r.json()["data"])
    return out


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 800) -> str:
    r = _client.post(
        f"{BASE}/chat/completions",
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
