"""FastAPI app: /upload, /chat, /reset, and a static chat UI at /."""
from __future__ import annotations
import os, tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import store as store_mod, ingest, query as qmod, search, generate, mistral

app = FastAPI(title="Mini RAG")


@app.exception_handler(Exception)
async def _unhandled(_req: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"answer": f"Server error: {type(exc).__name__}: {exc}", "citations": []})

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
            pages = ingest.extract_pages(path, f.filename)
            chunks = ingest.chunk_pages(pages, doc=f.filename)
            if not chunks:
                added.append({"file": f.filename, "chunks": 0, "warn": "no extractable text"})
                continue
            doc_meta = ingest.extract_doc_meta(pages, f.filename)
            embs = mistral.embed([c.text for c in chunks])
            replaced = _store.remove_doc(f.filename)
            _store.add(chunks, embs)
            _store.set_doc_meta(f.filename, doc_meta)
            entry = {"file": f.filename, "chunks": len(chunks), "title": doc_meta.get("title")}
            if replaced:
                entry["replaced"] = replaced
            added.append(entry)
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

    refusal = qmod.screen_for_sensitive_data(q)
    if refusal:
        return {"answer": refusal, "route": "refused", "citations": []}

    decision = qmod.route_query(q)
    route = decision["route"]

    if route == "chat":
        return {"answer": decision.get("response") or "", "route": "chat", "citations": []}

    if route == "clarify":
        return {"answer": decision.get("question") or "Could you clarify what you're asking?", "route": "clarify", "citations": []}

    if not _store.chunks:
        return {"answer": "The knowledge base is empty. Please upload PDFs first.", "route": "kb", "citations": []}

    rewritten = decision.get("rewrite") or q
    fmt = decision.get("format") or "qa"
    hits = search.retrieve(_store, rewritten, k=body.k)
    out = generate.answer(q, hits, fmt)
    out["route"] = "kb"
    out["rewritten_query"] = rewritten
    return out


@app.post("/reset")
def reset():
    global _store
    _store = store_mod.Store()
    store_mod.save(_store)
    return {"ok": True}


@app.delete("/docs")
def delete_doc(name: str):
    removed = _store.remove_doc(name)
    if removed == 0:
        raise HTTPException(404, f"No document named {name!r} in the KB")
    store_mod.save(_store)
    return {"removed_chunks": removed, "doc": name, "remaining": len(_store.chunks)}


@app.get("/stats")
def stats():
    docs = sorted({c.doc for c in _store.chunks})
    return {"docs": docs, "chunks": len(_store.chunks)}
