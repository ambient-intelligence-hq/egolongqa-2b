"""Acceptance gates for the vocab-pruned EgoLongQA junior.

EgoProactive's check 4 compared a single yes/no probability, which is all its task emitted. Our
junior GENERATES free text, so a pruned row is not just an input it cannot read -- it is an output
it can no longer produce. Equal accuracy would not prove equivalence (two different runs can score
the same by luck), so the gate here is PREDICTION equivalence, token by token:

  1. exhaustive remap coverage   -- every tokenizer id maps in-range (as EgoProactive)
  2. fallback round-trip         -- pruned ids decode back to identical text (as EgoProactive)
  3. unicode fuzz                -- CJK/Arabic/emoji/accents survive (as EgoProactive)
  4. argmax safety               -- on real prompts, was the original's top token ever PRUNED?
                                    if yes, that position is where generation must diverge
  5. logit equivalence           -- pruned logits == original logits on every KEPT row
  6. greedy generation identity  -- full completions, decoded, compared string-for-string

Checks 4-6 run on real eval70_shuf prompts with real frames, through the same prompt construction
as src/evaluate.py, so they test the shipped path rather than a synthetic probe.

Usage:
  python src/vocab_pruning/verify_egolongqa.py \
      --pruned work/pruned --orig work/merged --limit 12
"""
import argparse, json, os, re, sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruned", default="work/pruned")
    ap.add_argument("--orig", default="work/merged")
    ap.add_argument("--data", default="data/eval70_shuf.jsonl")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--num-frames", type=int, default=100)
    ap.add_argument("--max-pixels", type=int, default=50176)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--skip-gen", action="store_true", help="checks 1-5 only")
    ap.add_argument("--out", default="reports/verify_pruned_egolongqa.json")
    a = ap.parse_args()

    r = json.load(open(os.path.join(a.pruned, "vocab_remap.json")))
    KEEP, ADD, NADD, NNEW = r["keep_low"], r["added_start"], r["n_added"], r["n_new"]
    FB = {int(k): v for k, v in r["fallback"].items()}
    EXTRAS = [int(x) for x in r.get("extras", [])]
    EXTRA_POS = {t: KEEP + k for k, t in enumerate(EXTRAS)}
    ADDED_BASE = KEEP + len(EXTRAS)
    keep_ids = list(range(KEEP)) + EXTRAS + list(range(ADD, ADD + NADD))
    inv = keep_ids                                  # new_id -> old_id

    proc = AutoProcessor.from_pretrained(a.orig)
    tok = proc.tokenizer

    def new_of(i):
        if i < KEEP: return i
        if i in EXTRA_POS: return EXTRA_POS[i]
        if ADD <= i < ADD + NADD: return ADDED_BASE + (i - ADD)
        return None

    def remap(ids):
        out = []
        for t in ids:
            n = new_of(t)
            if n is None: out.extend(FB.get(t, [0]))
            else: out.append(n)
        return out

    report = {}

    # ---- 1. exhaustive coverage ----------------------------------------------------------
    bad = [i for i in range(len(tok))
           if (lambda m: not m or any(x < 0 or x >= NNEW for x in m))(remap([i]))]
    print(f"[1] coverage: {len(tok):,} ids, out-of-range: {len(bad)}")
    assert not bad, f"OUT OF RANGE: {bad[:10]}"
    report["coverage_out_of_range"] = len(bad)

    # ---- 2. fallback round-trip ----------------------------------------------------------
    mism = sum(1 for i in list(FB) if tok.decode([i]) != tok.decode(FB[i]))
    print(f"[2] fallback round-trip: {len(FB):,} pruned ids, text mismatches: {mism}")
    report["fallback_mismatches"] = mism

    # ---- 3. unicode fuzz -----------------------------------------------------------------
    fuzz = ["How do I sauté onions?", "中文测试 with English", "مرحبا بالعالم", "안녕하세요",
            "Привет мир", "emoji 😀🔥🎉 test", "naïve café résumé jalapeño", "ελληνικά", "עברית",
            "'curly' “quotes” — em-dash… ½ ¾ ± × ÷", "日本語のテスト",
            "González Pérez Hostel Aksaray"]
    worst = 0
    for s in fuzz:
        ids = tok.encode(s, add_special_tokens=False)
        m = remap(ids)
        assert all(0 <= x < NNEW for x in m), f"fuzz out of range: {s!r}"
        assert tok.decode([inv[x] for x in m]) == tok.decode(ids), f"fuzz text changed: {s!r}"
        worst = max(worst, len(m) - len(ids))
    print(f"[3] unicode fuzz: all in range, text preserved; worst growth +{worst} tokens")
    report["fuzz_worst_growth"] = worst

    # ---- real prompts --------------------------------------------------------------------
    try:
        from pathlib import Path as _P
        SYS = (_P(__file__).resolve().parents[2] / "prompts" / "junior_system.txt").read_text()
    except Exception:
        SYS = open(os.environ["PROMPT_FILE"]).read()
    recs = [json.loads(l) for l in open(a.data)][:a.limit]

    def build(rec):
        paths = json.load(open(Path(a.frames) / rec["video_id"] / "frames.json"))[:a.num_frames]
        q = ("Provide the detailed video description and answer to the question now: "
             f"Question:{rec['question']} {rec['mcq_options']}")
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": [{"type": "image"} for _ in paths]
                                            + [{"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = re.sub(r"<think>\s*</think>\s*", "", text)
        imgs = [Image.open(p).convert("RGB") for p in paths]
        return proc(text=[text], images=imgs, return_tensors="pt",
                    images_kwargs={"max_pixels": a.max_pixels, "min_pixels": 3136})

    print(f"\nloading both models ({len(recs)} real prompts, {a.num_frames} frames each)...")
    mo = AutoModelForImageTextToText.from_pretrained(a.orig, dtype=torch.bfloat16,
                                                     device_map="cuda", attn_implementation="sdpa").eval()
    mp = AutoModelForImageTextToText.from_pretrained(a.pruned, dtype=torch.bfloat16,
                                                     device_map="cuda", attn_implementation="sdpa").eval()

    kept_old = torch.tensor(keep_ids, dtype=torch.long, device="cuda")
    argmax_pruned_hits = 0
    max_abs_logit_diff = 0.0
    gen_identical = gen_total = 0
    letter_same = 0
    rows = []
    for k, rec in enumerate(recs):
        b = build(rec)
        d = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in b.items()}
        src = b["input_ids"][0].tolist()
        rid = remap(src)
        dp = dict(d)
        dp["input_ids"] = torch.tensor([rid], dtype=torch.long, device="cuda")
        dp["attention_mask"] = torch.ones((1, len(rid)), dtype=torch.long, device="cuda")

        with torch.no_grad():
            lo = mo(**d).logits[0, -1].float()
            lp = mp(**dp).logits[0, -1].float()

        # 4. was the original's top-1 a row we pruned? that is the only way prefill can diverge
        top_old = int(lo.argmax())
        if new_of(top_old) is None:
            argmax_pruned_hits += 1
        # 5. every KEPT row must carry the identical logit
        diff = float((lo.index_select(0, kept_old) - lp).abs().max())
        max_abs_logit_diff = max(max_abs_logit_diff, diff)

        row = {"index": rec["index"], "prompt_len_old": len(src), "prompt_len_new": len(rid),
               "top1_pruned": new_of(top_old) is None, "max_abs_logit_diff": diff}

        if not a.skip_gen:
            with torch.no_grad():
                go = mo.generate(**d, max_new_tokens=a.max_new_tokens, do_sample=False)
                gp = mp.generate(**dp, max_new_tokens=a.max_new_tokens, do_sample=False)
            to = tok.decode(go[0][len(src):], skip_special_tokens=True)
            tp = tok.decode([inv[int(x)] for x in gp[0][len(rid):]], skip_special_tokens=True)
            gen_total += 1
            same = (to == tp)
            gen_identical += same
            L = lambda t: (re.search(r"<choice>\s*([A-D])\s*</choice>", t or "", re.I) or [None, None])[1]
            letter_same += (L(to) == L(tp))
            row.update({"gen_identical": same, "letter_orig": L(to), "letter_pruned": L(tp),
                        "gen_len_old": len(go[0]) - len(src), "gen_len_new": len(gp[0]) - len(rid)})
            if not same:
                row["first_divergence"] = next((i for i, (x, y) in enumerate(zip(to, tp)) if x != y),
                                               min(len(to), len(tp)))
        rows.append(row)
        print(f"  [{k+1}/{len(recs)}] idx={rec['index']:>3} len {len(src)}->{len(rid)} "
              f"logitdiff={diff:.3e}"
              + ("" if a.skip_gen else
                 f" gen_identical={row['gen_identical']} letter {row['letter_orig']}/{row['letter_pruned']}"),
              flush=True)

    print(f"\n[4] argmax safety: original top-1 landed on a PRUNED row in "
          f"{argmax_pruned_hits}/{len(recs)} prompts")
    print(f"[5] logit equivalence over kept rows: max |diff| = {max_abs_logit_diff:.3e}")
    if not a.skip_gen:
        print(f"[6] greedy generation identical: {gen_identical}/{gen_total}   "
              f"same <choice> letter: {letter_same}/{gen_total}")
    report.update({"n_prompts": len(recs), "argmax_pruned_hits": argmax_pruned_hits,
                   "max_abs_logit_diff": max_abs_logit_diff,
                   "gen_identical": gen_identical, "gen_total": gen_total,
                   "letter_same": letter_same, "rows": rows})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")
    print("VERIFY_DONE")


main()
