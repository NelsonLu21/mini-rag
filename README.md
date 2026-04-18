# Mini RAG

A minimal Retrieval-Augmented Generation system over PDFs.
FastAPI backend, vanilla HTML chat UI, Mistral AI for OCR / embeddings / generation.
No external vector database, no RAG framework.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/NelsonLu21/mini-rag.git mini-rag
cd mini-rag
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure your Mistral API key

```bash
cp .env.example .env
# then edit .env and set MISTRAL_API_KEY=...
```

### 4. Run

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000**. Upload a PDF (or a folder of PDFs) and chat.

### Optional: evaluation

Two scripts under `test/` reproduce the small-scale evaluation against
[Vectara's `open_ragbench`](https://huggingface.co/datasets/vectara/open_ragbench):

```bash
# 20 papers + 100 questions; uploads to the running server, dumps a side-by-side report
python test/eval_ragbench.py

# LLM-as-judge grader: produces correct / partial / wrong / refused verdicts
python test/grade.py
```

Reports land under `test/ragbench/run_<timestamp>{.json,.md}` and `..._graded.{json,md}`.

---

## Design

Two HTTP endpoints carry the system: `POST /upload` ingests PDFs into a
persistent in-memory index, and `POST /chat` answers a user message
against that index.

### Upload workflow

```
PDF
 │ 
 ▼
Mistral OCR (mistral-ocr-latest)        per-page markdown + image refs
 │
 ├─ extract_doc_meta                    {title, authors, date, summary}
 │                                      one LLM call on first 2 pages
 ▼
chunk_pages                             list[Chunk]
   • split markdown into heading + paragraph blocks
   • pack paragraphs into ~900-char windows with 150-char overlap
   • flush at every heading boundary (no chunk crosses sections)
   • carry per-chunk heading path
   • inline image tags become [Image: alt] placeholders
 │
 ▼
mistral.embed(chunk texts)              1024-d float32 vectors, L2-normalized
 │
 ▼
Store.remove_doc(filename)              auto-replace existing version
Store.add(chunks, embeddings)
Store.set_doc_meta(filename, meta)
 │
 ▼
pickle to data/store.pkl                no third-party vector DB
```

The `Store` keeps chunks, embeddings, and per-document metadata as
parallel lists / dicts keyed by the same integer index. On startup
`load()` rehydrates from the pickle; older schema versions are tolerated
via `getattr`-guarded reads.

### Chat workflow

The handler is a guarded pipeline — each stage can short-circuit.

```
message
 │
 ▼
screen_for_sensitive_data               LLM JSON filter
   refuses extraction of SSN, bank/IBAN, card numbers,
   passwords/API keys, passport/license numbers.
   Routine contact info (email/phone/address) passes through.
 │  (pass)
 ▼
route_query                             LLM JSON router, one of:
   • "chat"     pleasantries OR meta-questions about the tool
                → return canned reply
   • "clarify"  vague / ambiguous message
                → return clarifying question
   • "kb"       default for anything with information content; produces:
                  rewrite (keyword-rich search query)
                  format  ("qa" | "list" | "table")
 │  (route == kb, KB non-empty)
 ▼
search.retrieve(rewrite, k=6)           top-k by cosine similarity over all chunks
 │
 ▼
generate.answer
   • gate: top-1 cosine < 0.55  → "insufficient evidence" (no LLM call)
   • prompt: base instructions + format template + numbered chunk context
   • mistral.chat → answer with inline [n] citations
   • post-hoc: keep only chunks the model actually cited; renumber [n] 1..m
   • evidence check: token-overlap + embedding cosine per answer sentence;
     flag unsupported claims
 │
 ▼
JSON response
   { answer, route, rewritten_query,
     citations[doc, page, headings, similarity, text],
     evidence_check.unsupported[] }
```

The UI renders inline `[n]` as clickable anchors and groups citations by
`doc p.N [1] [2]`. Clicking a pill (or an inline `[n]`) toggles a panel
showing the full chunk text.

### Key design decisions

- **Mistral OCR over plain text extraction.** PDFs are visually structured.
  Plain-text extractors mangle tables, columns, and line breaks;
  OCR returns clean markdown with headings preserved.
- **Heading-aware paragraph chunking.** Paragraphs are packed into
  ~900-char windows with 150-char overlap, and headings treated as hard chunk
  boundaries. Each chunk carries its full heading path, used both in the LLM 
  context and in citation labels.
- **One LLM call for routing, rewriting, and format.** A single call per query.
  Returns the route (chat / kb / clarify), a keyword-rich rewrite,
  and the answer format (qa / list / table). The rewrite prompt asks for
  synonyms and words that are likely to appear together in the chunks 
  (e.g. "authors", "affiliation") so keyword-poor
  user phrasing still surfaces the right chunk.
- **Cosine with a hard refusal gate.** All embeddings are L2-normalized at
  insertion, so retrieval is one matmul and no vector DB is required.
  Cosine has a stable, interpretable scale, so a single threshold
  (`top-1 < 0.55 → "insufficient evidence"`) deterministically catches
  off-topic queries before the LLM is even called.
- **Cited-only sources, renumbered.** After generation we parse the
  answer's `[n]` markers, drop chunks the model didn't actually cite, and
  renumber the remainder so the displayed Sources list is `[1]..[m]` with
  no gaps. Users only see references that are used in the answer.
- **Two-stage evidence check.** Every answer sentence is first matched
  against retrieved chunks by token overlap (large overlap almost certainly 
  means the answer is grounded); only suspicious sentences get embedded for 
  a cosine check. 

---

## Validation

We exercised the system against
[Vectara's `open_ragbench`](https://huggingface.co/datasets/vectara/open_ragbench),
an arXiv-derived benchmark whose QA pairs are labelled with the source
document each answer comes from.

### Setup
- **20 papers**, **100 text-only queries**, round-robin across papers
  (the grader tools live at `test/eval_ragbench.py` and `test/grade.py`).
- Two independent checks per query:
  1. **Citation correctness** (pure Python) — does any citation we returned point at the golden paper?
  2. **Answer verdict** (Mistral LLM-as-judge in JSON mode) — `correct` / `partial` / `wrong` / `refused`.

### Results (100 queries)

| Metric | Result |
|---|---|
| Correct answer | **89 / 100** (89%) |
| Partial answer | 7 / 100 (7%) |
| Wrong answer | 2 / 100 (2%) |
| Refused | 2 / 100 (2%) |
| Citation includes the golden doc | **96 / 100** (96%) |
| Both correct **and** cited the right doc | **88 / 100** (88%) |

### Notes
- Both refusals were the **right** behaviour: the queries asked about content
  not present in the golden paper, and the 0.55 cosine gate held the line
  rather than guessing.
- One of the two `wrong` cases was a routed-to-`clarify` decision on a
  borderline-ambiguous question; the other was a LaTeX-heavy physics action
  where our answer described a similar-but-different functional form.
- The 7 partials were mild omissions — core claim correct, but a nuance was
  missing or extra context was added.

Per-run artefacts are saved under
`test/ragbench/run_<timestamp>.{json,md}` and `..._graded.{json,md}`.
