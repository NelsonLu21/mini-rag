"""Small-scale evaluation against Vectara's open_ragbench dataset.

Pipeline:
  1. Download the 4 metadata JSONs (a few MB) from HuggingFace.
  2. Pick N papers with the most text-only queries.
  3. Download just those N PDFs from the arXiv URLs.
  4. Upload them to the running server (/upload).
  5. Run the queries through /chat and dump a side-by-side comparison.

Usage (server must be running on http://localhost:8000):
  python test/eval_ragbench.py                          # default 3 papers, 10 queries
  python test/eval_ragbench.py --n-papers 5 --n-queries 20
  python test/eval_ragbench.py --reset                  # wipe KB first
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import httpx

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent / "ragbench"
META_DIR = ROOT / "meta"
PDF_DIR = ROOT / "pdfs"
HF_BASE = "https://huggingface.co/datasets/vectara/open_ragbench/resolve/main/pdf/arxiv"
SERVER = os.environ.get("RAG_SERVER", "http://localhost:8000")
META_FILES = ("queries.json", "qrels.json", "answers.json", "pdf_urls.json")


def download_metadata() -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    for name in META_FILES:
        p = META_DIR / name
        if p.exists() and p.stat().st_size > 0:
            continue
        url = f"{HF_BASE}/{name}"
        print(f"  fetch {name} ...", end="", flush=True)
        r = httpx.get(url, follow_redirects=True, timeout=120.0)
        r.raise_for_status()
        p.write_bytes(r.content)
        print(f" {len(r.content)//1024} KB")


def load_metadata() -> dict:
    return {name.split(".")[0]: json.loads((META_DIR / name).read_text(encoding="utf-8"))
            for name in META_FILES}


def select_subset(meta: dict, n_papers: int) -> list[tuple[str, list[str]]]:
    text_queries = {qid for qid, q in meta["queries"].items() if q.get("source") == "text"}
    by_paper: dict[str, list[str]] = {}
    for qid, rel in meta["qrels"].items():
        if qid not in text_queries:
            continue
        doc = rel["doc_id"]
        if doc not in meta["pdf_urls"]:
            continue
        by_paper.setdefault(doc, []).append(qid)
    ranked = sorted(by_paper.items(), key=lambda x: -len(x[1]))
    return ranked[:n_papers]


def download_pdfs(selected: list[tuple[str, list[str]]], pdf_urls: dict) -> dict[str, Path]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for paper_id, _ in selected:
        safe_name = paper_id.replace("/", "_") + ".pdf"
        p = PDF_DIR / safe_name
        if not (p.exists() and p.stat().st_size > 1000):
            url = pdf_urls[paper_id]
            print(f"  fetch {paper_id} ...", end="", flush=True)
            r = httpx.get(url, follow_redirects=True, timeout=180.0)
            r.raise_for_status()
            p.write_bytes(r.content)
            print(f" {len(r.content)//1024} KB")
        paths[paper_id] = p
    return paths


def server_reset() -> None:
    r = httpx.post(f"{SERVER}/reset", timeout=30.0)
    r.raise_for_status()


def server_upload_one(path: Path) -> dict:
    files = [("files", (path.name, path.read_bytes(), "application/pdf"))]
    r = httpx.post(f"{SERVER}/upload", files=files, timeout=600.0)
    r.raise_for_status()
    return r.json()


def server_chat(message: str) -> dict:
    r = httpx.post(f"{SERVER}/chat", json={"message": message}, timeout=180.0)
    r.raise_for_status()
    return r.json()


def check_server() -> None:
    try:
        r = httpx.get(f"{SERVER}/stats", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: cannot reach server at {SERVER}. Start it with `uvicorn app.main:app` first.")
        print(f"  ({type(e).__name__}: {e})")
        sys.exit(1)


def truncate(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-papers", type=int, default=20)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--reset", action="store_true", help="Reset KB before uploading")
    ap.add_argument("--skip-upload", action="store_true", help="Assume PDFs are already in the KB")
    args = ap.parse_args()

    print("=== open_ragbench mini-eval ===\n")
    check_server()

    print("[1/5] metadata")
    download_metadata()
    meta = load_metadata()
    print(f"  queries={len(meta['queries'])}  qrels={len(meta['qrels'])}  papers={len(meta['pdf_urls'])}")

    print(f"\n[2/5] selecting top {args.n_papers} text-only-heavy papers")
    selected = select_subset(meta, args.n_papers)
    for pid, qids in selected:
        print(f"  {pid}  ({len(qids)} text queries)")

    print(f"\n[3/5] downloading {len(selected)} PDFs")
    paths = download_pdfs(selected, meta["pdf_urls"])

    if args.reset:
        print("\n[4a] resetting KB")
        server_reset()

    if args.skip_upload:
        print("\n[4/5] skipping upload")
    else:
        print(f"\n[4/5] uploading {len(paths)} PDFs to {SERVER} (one at a time)")
        for i, (pid, p) in enumerate(paths.items(), 1):
            t0 = time.time()
            try:
                up = server_upload_one(p)
                item = (up.get("ingested") or [{}])[0]
                dt = time.time() - t0
                print(f"  [{i}/{len(paths)}] {p.name}: {item.get('chunks', 0)} chunks  title={item.get('title')!r}  ({dt:.0f}s)")
            except Exception as e:
                print(f"  [{i}/{len(paths)}] {p.name}: FAILED ({type(e).__name__}: {e})")

    queries: list[tuple[str, str]] = []
    iterators = [iter(qids) for _, qids in selected]
    while len(queries) < args.n_queries:
        added = False
        for i, it in enumerate(iterators):
            qid = next(it, None)
            if qid is None:
                continue
            queries.append((qid, selected[i][0]))
            added = True
            if len(queries) >= args.n_queries:
                break
        if not added:
            break

    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = ROOT / f"run_{stamp}.json"
    md_path = ROOT / f"run_{stamp}.md"
    print(f"\n[5/5] running {len(queries)} queries")
    print(f"  incrementally saving to {json_path}")
    results: list[dict] = []
    for i, (qid, pid) in enumerate(queries, 1):
        q = meta["queries"][qid]["query"]
        gt = meta["answers"].get(qid, "")
        print(f"\n--- [{i}/{len(queries)}] paper={pid} qid={qid}")
        print(f"Q:  {truncate(q, 200)}")
        print(f"GT: {truncate(gt, 200)}")
        try:
            resp = server_chat(q)
        except Exception as e:
            print(f"ERR {type(e).__name__}: {e}")
            results.append({"qid": qid, "paper": pid, "query": q, "gt": gt,
                            "route": None, "answer": None, "error": f"{type(e).__name__}: {e}"})
            json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            continue
        ans = resp.get("answer", "")
        route = resp.get("route")
        cites = [f"{c['doc']}:{c['page']}" for c in (resp.get("citations") or [])]
        print(f"RAG ({route}): {truncate(ans, 400)}")
        if cites:
            print(f"    cites: {', '.join(cites)}")
        results.append({
            "qid": qid, "paper": pid, "query": q, "gt": gt,
            "route": route, "answer": ans,
            "rewritten": resp.get("rewritten_query"),
            "citations": resp.get("citations", []),
            "unsupported": (resp.get("evidence_check") or {}).get("unsupported", []),
        })
        json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path.write_text(_render_markdown(results), encoding="utf-8")
    print(f"\nSaved {len(results)} results:")
    print(f"  JSON: {json_path}")
    print(f"  Side-by-side markdown: {md_path}")


def _md_esc(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def _render_markdown(results: list[dict]) -> str:
    lines = [
        "# RAG eval — side-by-side vs ground truth",
        "",
        f"Total queries: **{len(results)}**",
        "",
    ]
    for i, r in enumerate(results, 1):
        cites = ", ".join(f"{c['doc']} p.{c['page']}" for c in r.get("citations") or [])
        lines.append(f"## {i}. `{r['paper']}` · route=`{r.get('route')}`")
        lines.append("")
        lines.append(f"**Query:** {_md_esc(r['query'])}")
        lines.append("")
        lines.append("| Ground truth | RAG answer |")
        lines.append("|---|---|")
        lines.append(f"| {_md_esc(r.get('gt', ''))} | {_md_esc(r.get('answer', ''))} |")
        lines.append("")
        if r.get("rewritten"):
            lines.append(f"*rewritten query:* `{_md_esc(r['rewritten'])}`  ")
        if cites:
            lines.append(f"*citations:* {cites}  ")
        if r.get("unsupported"):
            lines.append(f"*unsupported claims:* {len(r['unsupported'])}  ")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
