#!/usr/bin/env python3
"""Build a distillation SFT dataset to fine-tune Qwen3.5-2B as a JUNIOR PERCEIVER
(not as the senior tool-using agent).

The junior's contract (ambient/tools/preempt_answer.py + PREMTPT_FOR_ANSWER_TOOL_PROMPT):
  INPUT   : ~100 UNIFORM frames of the whole video (max_dim 768) + the MCQ question
  OUTPUT  : <video_description>timestamped overview</video_description>
            <answer><choice>X</choice><reason>..</reason><citation>..</citation></answer>

That input protocol is EXACTLY what our 2B student already gets at inference, which is why
junior traces (not the senior's final answer) are the right supervision.

Teacher traces are recovered from the agentic runs' agent_trajectory.json: the senior's context
message embeds each independent junior pass as "===== Junior agent N =====" blocks. Every run
has ~3 junior passes per sample -> natural augmentation. Only passes whose <choice> == ground
truth are kept.

Reads run dirs READ-ONLY; writes only to --out-dir.

Usage:
  python src/build_distill_dataset.py --out-dir data
"""
import argparse, os, json, glob, re
from pathlib import Path

VAL = os.environ.get("EGOLONGQA_VAL",
                     "data/wearable_ai_2026_egolongqa_val_700.jsonl")

TEACHER_RUNS = {
    "qwen122b_700":     "benchmark_outputs_qwen122b_700",
    "gemma_630":        "benchmark_outputs_gemma_junior_630",
    "gate122bfp8_400f": "benchmark_outputs_gate122bfp8_400f",
    "gate27b_700":      "benchmark_outputs_gate27b_700",
    "gate35b_400f":     "benchmark_outputs_gate35b_400f_enriched",
    "gate35b_100f":     "benchmark_outputs_gate35b_100f_resample",
    "27b_junior":       "benchmark_outputs_27b_junior",
    "gemma_70":         "benchmark_outputs_gemma_junior",
}

JUNIOR_SPLIT = re.compile(r"=====\s*Junior agent\s*\d+\s*=====")
OVERVIEW_RE = re.compile(
    r"Approximate timestamped overview of the video.*?:\s*(.*?)\s*(?:\n\s*Full Video Duration:|\Z)",
    re.S)
CAND_RE = re.compile(r"Candidate answer from this pass:\s*(.*)", re.S)
CHOICE_RE = re.compile(r"<choice>\s*([A-D])\s*</choice>", re.I)
REASON_RE = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.S | re.I)
CITE_RE = re.compile(r"<citation>\s*(.*?)\s*</citation>", re.S | re.I)
# a junior overview must not narrate tool use (juniors never call tools; guards against
# accidentally scraping a senior/focus-tool block)
TOOL_LEAK = re.compile(r"\b(search_clip|focus_clip|I called|tool_result)\b", re.I)
# some teachers dumped raw chain-of-thought into <video_description> instead of a clean
# narration ("The user wants ...", "I need to ..."). Those are 10-20x longer than a real
# overview and teach the WRONG output shape — drop them.
META_COT = re.compile(r"^\s*(the user (wants|is asking)|i need to|let me|okay,|first,? i)", re.I)
MAX_OVERVIEW_CHARS = 5000   # real overviews are ~600-2500 chars; longer = leaked CoT


def junior_blocks(traj_path):
    """Yield raw '===== Junior agent N =====' text blocks from a trajectory file."""
    try:
        d = json.load(open(traj_path))
    except Exception:
        return
    seen = set()
    for turn in (d.get("turns") or []):
        for m in (turn.get("messages") or []):
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            c = m.get("content")
            if not isinstance(c, list):
                continue
            for b in c:
                if not (isinstance(b, dict) and b.get("type") == "text"):
                    continue
                t = b.get("text") or ""
                if "Junior agent" not in t:
                    continue
                for blk in JUNIOR_SPLIT.split(t)[1:]:
                    key = blk[:200]
                    if key not in seen:
                        seen.add(key)
                        yield blk


def parse_junior(blk):
    """-> {overview, choice, reason, citation} or None."""
    mo = OVERVIEW_RE.search(blk)
    mc = CAND_RE.search(blk)
    if not mo or not mc:
        return None
    overview = mo.group(1).strip()
    if len(overview) < 80 or TOOL_LEAK.search(overview):
        return None
    if len(overview) > MAX_OVERVIEW_CHARS or META_COT.search(overview):
        return None
    cand = mc.group(1)
    ch = CHOICE_RE.search(cand)
    if not ch:
        return None
    rs, ct = REASON_RE.search(cand), CITE_RE.search(cand)
    return {"overview": overview, "choice": ch.group(1).upper(),
            "reason": (rs.group(1).strip() if rs else ""),
            "citation": (ct.group(1).strip() if ct else "")}


def target_text(p):
    """Reconstruct the junior's native output format (what the student must learn to emit)."""
    ans = f"<answer>\n<choice>{p['choice']}</choice>\n<reason>{p['reason']}</reason>\n"
    if p["citation"]:
        ans += f"<citation>{p['citation']}</citation>\n"
    ans += "</answer>"
    return f"<video_description>\n{p['overview']}\n</video_description>\n\n{ans}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--num-frames", type=int, default=100)
    ap.add_argument("--max-traces", type=int, default=4,
                    help="cap traces kept per sample (dedup/augmentation limit)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val) if l.strip()]
    v2i = {r["video_path"].split(".")[0]: i for i, r in enumerate(rows)}
    out = Path(args.out_dir)
    split = json.load(open(out / "split.json"))
    train, held = set(split["train"]), set(split["heldout"])

    by_idx, stats = {}, {}
    for name, root in TEACHER_RUNS.items():
        kept = wrong = bad = 0
        for f in glob.glob(f"{root}/run_*/samples/*/agent_trajectory.json") + \
                 glob.glob(f"{root}/samples/*/agent_trajectory.json"):
            vid = Path(f).parent.name.split("_", 1)[-1]
            i = v2i.get(vid)
            if i is None:
                continue
            gt = str(rows[i]["mcq_answer"]).upper()[:1]
            for blk in junior_blocks(f):
                p = parse_junior(blk)
                if p is None:
                    bad += 1
                    continue
                if p["choice"] != gt:          # keep only CORRECT junior passes
                    wrong += 1
                    continue
                by_idx.setdefault(i, []).append({"teacher": name, **p})
                kept += 1
        stats[name] = (kept, wrong, bad)

    n_tr = n_ho = rows_tr = 0
    ftr = open(out / "train_junior.jsonl", "w")
    fho = open(out / "heldout_junior.jsonl", "w")
    for i, traces in sorted(by_idx.items()):
        r = rows[i]
        # dedupe identical overviews, cap per sample
        seen, uniq = set(), []
        for t in traces:
            k = t["overview"][:300]
            if k not in seen:
                seen.add(k); uniq.append(t)
            if len(uniq) >= args.max_traces:
                break
        rec = {"index": i, "video_id": r["video_path"].split(".")[0],
               "question": r["question"], "mcq_options": r["mcq_options"],
               "answer": str(r["mcq_answer"]).upper()[:1], "category": r.get("category"),
               "frame_protocol": {"mode": "uniform", "num_frames": args.num_frames, "max_dim": 768},
               "traces": [{"teacher": t["teacher"], "target": target_text(t)} for t in uniq]}
        if i in train:
            ftr.write(json.dumps(rec) + "\n"); n_tr += 1; rows_tr += len(uniq)
        elif i in held:
            fho.write(json.dumps(rec) + "\n"); n_ho += 1
    ftr.close(); fho.close()

    print("junior passes (only CORRECT ones kept):")
    for k, (kept, wrong, bad) in stats.items():
        print(f"  {k:18s} kept={kept:5d}  dropped_wrong_choice={wrong:5d}  unparsable={bad}")
    print(f"\nTRAIN   samples={n_tr}/{len(train)}  trace-augmented rows={rows_tr}")
    print(f"HELDOUT samples={n_ho}/{len(held)}  (eval only — NEVER train on these)")
    print(f"-> {out}/train_junior.jsonl, {out}/heldout_junior.jsonl")


if __name__ == "__main__":
    main()
