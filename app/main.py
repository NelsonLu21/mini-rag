"""FastAPI app: /upload, /chat, /reset, and a static chat UI at /."""
from __future__ import annotations
import os, tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import store as store_mod, ingest, query as qmod, search, generate, mistral

app = FastAPI(title="Mini RAG")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_store = store_mod.load()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    added = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"{f.filename}: only .pdf supported")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await f.read())
            path = tmp.name
        try:
            pages = ingest.extract_pages(path)
            chunks = ingest.chunk_pages(pages, doc=f.filename)
            if not chunks:
                added.append({"file": f.filename, "chunks": 0, "warn": "no extractable text"})
                continue
            embs = mistral.embed([c.text for c in chunks])
            _store.add(chunks, embs)
            added.append({"file": f.filename, "chunks": len(chunks)})
        finally:
            os.unlink(path)
    store_mod.save(_store)
    return {"ingested": added, "total_chunks": len(_store.chunks)}


class ChatIn(BaseModel):
    message: str
    k: int = 6


@app.post("/chat")
def chat(body: ChatIn):
    q = body.message.strip()
    if not q:
        raise HTTPException(400, "empty message")

    # 1. policy refusal (PII / legal / medical)
    refusal = qmod.policy_check(q)
    if refusal:
        return {"answer": refusal, "intent": "refused", "citations": []}

    # 2. intent detection — skip retrieval for chit-chat
    intent = qmod.classify_intent(q)
    if intent == "greeting":
        return {
            "answer": "Hi! Upload PDFs and ask me about them.",
            "intent": intent,
            "citations": [],
        }

    if not _store.chunks:
        return {
            "answer": "The knowledge base is empty. Please upload PDFs first.",
            "intent": intent,
            "citations": [],
        }

    # 3. query rewrite for retrieval
    rewritten = qmod.transform(q)

    # 4. hybrid search
    hits = search.hybrid(_store, rewritten, k=body.k)

    # 5. generate answer with citations + evidence check
    out = generate.answer(q, hits, intent)
    out["intent"] = intent
    out["rewritten_query"] = rewritten
    return out


@app.post("/reset")
def reset():
    global _store
    _store = store_mod.Store()
    store_mod.save(_store)
    return {"ok": True}


@app.get("/stats")
def stats():
    docs = sorted({c.doc for c in _store.chunks})
    return {"docs": docs, "chunks": len(_store.chunks)}
