"""Capture the ORIGINAL model's generated token ids, on THIS machine, for the keep-set.

Why this exists rather than reusing an old eval dump: on the input side a pruned token byte-expands
and the text survives exactly, but on the OUTPUT side there is no fallback -- if the model's argmax
is a row we removed, it emits a different word entirely (we observed ' rame' -> ' bowl'). So the
output-side keep-set has to be built from tokens this model actually generates.

It must also be captured on the machine that will serve, because generation is only reproducible
per-machine here: a keep-set captured on one GPU did not cover what another GPU actually
not cover what this one emits, which is exactly how a pruned-argmax slipped through.

Greedy decoding means the generated id IS the argmax at every step, so counting high-ID generated
tokens measures the divergence risk directly -- no extra instrumentation needed.

Usage:
  python src/vocab_pruning/capture_output_vocab.py --data data/eval70_shuf.jsonl \
      --out work/genvocab.json
"""
import argparse, collections, json, os, re, sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="work/merged")
    ap.add_argument("--data", default="data/eval70_shuf.jsonl")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--num-frames", type=int, default=100)
    ap.add_argument("--max-pixels", type=int, default=50176)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--keep-low", type=int, default=143000)
    ap.add_argument("--added-start", type=int, default=248044)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thinking", action="store_true", default=False,
                    help="Qwen3.5 thinking mode. The keep-set must cover whichever mode SHIPS: "
                         "the container's 75.7% baseline ran --thinking, so its generations (and "
                         "therefore its rare tokens) differ from the non-thinking run.")
    ap.add_argument("--out", default="work/genvocab.json")
    a = ap.parse_args()

    try:
        from pathlib import Path as _P
        SYS = (_P(__file__).resolve().parents[2] / "prompts" / "junior_system.txt").read_text()
    except Exception:
        SYS = open(os.environ["PROMPT_FILE"]).read()

    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    recs = [json.loads(l) for l in open(a.data)]
    if a.limit:
        recs = recs[:a.limit]

    high = collections.Counter()
    n_gen_tokens = 0
    items_with_high = 0
    rows = []
    for k, rec in enumerate(recs):
        man = Path(a.frames) / rec["video_id"] / "frames.json"
        if not man.exists():
            continue
        paths = json.load(open(man))[:a.num_frames]
        paths = [q if Path(q).exists()
                 else str(Path(a.frames) / rec["video_id"] / Path(q).name) for q in paths]
        q = ("Provide the detailed video description and answer to the question now: "
             f"Question:{rec['question']} {rec['mcq_options']}")
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": [{"type": "image"} for _ in paths]
                                            + [{"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        **({"enable_thinking": True} if a.thinking else {}))
        if not a.thinking:
            text = re.sub(r"<think>\s*</think>\s*", "", text)
        b = proc(text=[text], images=[Image.open(p).convert("RGB") for p in paths],
                 return_tensors="pt",
                 images_kwargs={"max_pixels": a.max_pixels, "min_pixels": 3136}).to(model.device)
        with torch.no_grad():
            g = model.generate(**b, max_new_tokens=a.max_new_tokens, do_sample=False)
        gen = g[0][b["input_ids"].shape[1]:].tolist()
        n_gen_tokens += len(gen)
        hits = [t for t in gen if a.keep_low <= t < a.added_start]
        high.update(hits)
        items_with_high += bool(hits)
        comp = tok.decode(gen, skip_special_tokens=True)
        rows.append({"index": rec["index"], "n_gen": len(gen), "n_high": len(hits),
                     "high": sorted(set(hits)),
                     "letter": (re.search(r"<choice>\s*([A-D])\s*</choice>", comp, re.I)
                                or [None, None])[1],
                     "gt": rec.get("answer")})
        print(f"  [{k+1}/{len(recs)}] idx={rec['index']:>3} gen={len(gen):>4} "
              f"high-ID={len(hits)}"
              + (f"  {[tok.decode([t]) for t in sorted(set(hits))]}" if hits else ""), flush=True)

    print(f"\nitems whose generation touches a prunable row: {items_with_high}/{len(rows)}")
    print(f"generated tokens: {n_gen_tokens:,}   high-ID occurrences: {sum(high.values()):,} "
          f"({100*sum(high.values())/max(1,n_gen_tokens):.3f}%)   unique: {len(high)}")
    if high:
        print("top:", [(tok.decode([t]), c) for t, c in high.most_common(12)])
    json.dump({"model": a.model, "data": a.data, "n_items": len(rows),
               "items_with_high": items_with_high, "n_gen_tokens": n_gen_tokens,
               "unique_high": len(high), "occurrences": sum(high.values()),
               "high_ids": sorted(high), "counts": {str(t): c for t, c in high.most_common()},
               "decoded": {str(t): tok.decode([t]) for t in sorted(high)},
               "rows": rows}, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


main()
