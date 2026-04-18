"""Grade a RAG eval run.

Reads the most recent run_*.json produced by eval_ragbench.py, then:
  1. Python check — does any citation include the golden doc id?
  2. Mistral-judged verdict on whether the RAG answer agrees with the ground truth.

Writes <run>_graded.json and <run>_graded.md beside the source file.

Usage:
  python test/grade.py                         # grade the latest run
  python test/grade.py path/to/run_xxx.json    # grade a specific run
"""
from __future__ import annotations
import json, sys, time
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.mistral import chat as mistral_chat  # noqa: E402

ROOT = Path(__file__).parent / "ragbench"

JUDGE_SYS = """You are grading a Retrieval-Augmented Generation system's answer against a ground-truth answer.

Verdict options (pick exactly one):
- "correct": the system answer conveys the same core information as the ground truth, even if worded differently or more verbose. Extra context is fine.
- "partial": some of the system answer matches the ground truth, but something important is missing, contradictory, or wrong.
- "wrong": the system answer contradicts the ground truth, or answers a completely different question.
- "refused": the system declined to answer (e.g. "I don't have enough information", "insufficient evidence"). Use this category even when a refusal would have been wrong — judging uses the answer's effect, not intent.

Output JSON ONLY:
{"verdict": "correct" | "partial" | "wrong" | "refused", "reason": "one short sentence"}
"""


def judge(query: str, gt: str, answer: str) -> dict:
    if not (answer or "").strip():
        return {"verdict": "refused", "reason": "empty answer"}
    a_lower = answer.lower()
    if "insufficient evidence" in a_lower or "don't have enough information" in a_lower:
        return {"verdict": "refused", "reason": "system refused to answer"}
    user = f"Query: {query}\n\nGround truth: {gt}\n\nSystem answer: {answer}"
    try:
        raw = mistral_chat(
            [{"role": "system", "content": JUDGE_SYS},
             {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        data = json.loads(raw)
    except Exception as e:
        return {"verdict": "error", "reason": f"{type(e).__name__}: {e}"}
    v = data.get("verdict") if isinstance(data, dict) else None
    if v not in {"correct", "partial", "wrong", "refused"}:
        return {"verdict": "error", "reason": f"invalid verdict from judge: {v!r}"}
    return {"verdict": v, "reason": str(data.get("reason") or "").strip()}


def cite_correct(result: dict) -> bool:
    golden = (result.get("paper") or "").strip()
    if not golden:
        return False
    for c in result.get("citations") or []:
        if golden in c.get("doc", ""):
            return True
    return False


def latest_run() -> Path:
    files = sorted(ROOT.glob("run_*.json"))
    files = [f for f in files if "_graded" not in f.stem]
    if not files:
        print(f"No run_*.json found in {ROOT}. Run eval_ragbench.py first.")
        sys.exit(1)
    return files[-1]


def _md_esc(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    print(f"Grading {path}")
    results = json.loads(path.read_text(encoding="utf-8"))
    n = len(results)
    graded: list[dict] = []
    t0 = time.time()
    for i, r in enumerate(results, 1):
        cite_ok = cite_correct(r)
        j = judge(r.get("query", ""), r.get("gt", ""), r.get("answer", ""))
        graded.append({**r, "cite_correct": cite_ok, "judge": j})
        print(f"[{i}/{n}] verdict={j['verdict']:<9} cite={'OK ' if cite_ok else 'no '}  {(r.get('query') or '')[:90]}")

    verdicts = Counter(g["judge"]["verdict"] for g in graded)
    cite_hits = sum(1 for g in graded if g["cite_correct"])
    both = sum(1 for g in graded if g["judge"]["verdict"] == "correct" and g["cite_correct"])

    print()
    print("=" * 60)
    print(f"Total queries:   {n}   (elapsed {time.time() - t0:.0f}s)")
    for v in ("correct", "partial", "wrong", "refused", "error"):
        c = verdicts.get(v, 0)
        print(f"  {v:<8}: {c:>3}  ({c * 100 // n:>3}%)")
    print(f"Citation includes golden doc: {cite_hits}/{n}  ({cite_hits * 100 // n}%)")
    print(f"BOTH correct answer & correct cite: {both}/{n}  ({both * 100 // n}%)")

    out_json = path.with_name(path.stem + "_graded.json")
    out_json.write_text(json.dumps(graded, indent=2, ensure_ascii=False), encoding="utf-8")

    md: list[str] = []
    md.append("# RAG eval — graded")
    md.append("")
    md.append(f"Source: `{path.name}`")
    md.append("")
    md.append(f"- Total queries: **{n}**")
    for v in ("correct", "partial", "wrong", "refused", "error"):
        c = verdicts.get(v, 0)
        md.append(f"- {v}: **{c}** ({c * 100 // n}%)")
    md.append(f"- Citation includes golden doc: **{cite_hits}/{n}** ({cite_hits * 100 // n}%)")
    md.append(f"- Correct answer AND correct citation: **{both}/{n}** ({both * 100 // n}%)")
    md.append("")
    md.append("## Per-query verdicts")
    md.append("")
    md.append("| # | verdict | cite | paper | query |")
    md.append("|---|---|---|---|---|")
    for i, g in enumerate(graded, 1):
        md.append(f"| {i} | {g['judge']['verdict']} | {'OK' if g['cite_correct'] else '—'} | {g['paper']} | {_md_esc(g.get('query',''))[:120]} |")
    md.append("")
    md.append("## Wrong / partial answers")
    md.append("")
    for i, g in enumerate(graded, 1):
        v = g["judge"]["verdict"]
        if v not in ("wrong", "partial"):
            continue
        md.append(f"### {i}. `{v}` · paper `{g['paper']}` · cite {'OK' if g['cite_correct'] else 'wrong'}")
        md.append("")
        md.append(f"**Query:** {_md_esc(g.get('query',''))}")
        md.append("")
        md.append(f"**Ground truth:** {_md_esc(g.get('gt',''))}")
        md.append("")
        md.append(f"**RAG answer:** {_md_esc(g.get('answer',''))}")
        md.append("")
        md.append(f"**Judge reason:** {g['judge']['reason']}")
        md.append("")
    out_md = path.with_name(path.stem + "_graded.md")
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"\nSaved:\n  JSON: {out_json}\n  MD:   {out_md}")


if __name__ == "__main__":
    main()
