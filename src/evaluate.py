#!/usr/bin/env python
"""Evaluate the junior-perceiver 2B (base vs LoRA-tuned) on the held-out set.

Runs the SAME contract the model was trained on:
  system(PREMTPT_FOR_ANSWER_TOOL_PROMPT) + user([N uniform frames] + question)
    -> <video_description>..</video_description><answer><choice>X</choice>..</answer>

IMPORTANT: pass the same --max-pixels used for training, and run the BASE model at that
same resolution as the control, so the measured delta is the fine-tuning effect and not a
resolution change (our 64.3% baseline was measured at 768px / ~437 tok/frame).

Usage:
  # tuned
  python src/evaluate.py --adapter runs/armB/checkpoint-367
  # base control (same resolution)
  python src/evaluate.py
"""
import argparse, json, re, sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, str(Path(__file__).resolve().parent))   # answer_fallback.py sits beside this
from answer_fallback import needs_fallback, fallback_choice
import os as _os
# --- system prompt -----------------------------------------------------------
# Loaded from prompts/junior_system.txt rather than imported, so this repo has no
# dependency on the wider agent codebase. The md5 is asserted because the prompt
# carries TWO TRAILING SPACES: a version with them stripped (len 1998) changed
# 10 of 70 held-out answers while leaving total accuracy unchanged.
_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "junior_system.txt"
JUNIOR_SYSTEM = _PROMPT.read_text()
import hashlib as _h
assert _h.md5(JUNIOR_SYSTEM.encode()).hexdigest() == "f342eb86cceb9ee18c9fba0ceae9f013", (
    f"junior_system.txt is not the shipped prompt (len={len(JUNIOR_SYSTEM)}, expected 2000). "
    "Check your editor has not stripped trailing whitespace.")

# STUDENT_MODEL lets the eval point at a MERGED checkpoint (adapter folded in) so the
# submission container can be verified against the exact same harness that produced
# our reported numbers, rather than a re-implementation.
MODEL = _os.environ.get("STUDENT_MODEL", "Qwen/Qwen3.5-2B")
CHOICE_RE = re.compile(r"<choice>\s*([A-D])\s*</choice>", re.I)
LOOSE_RE = re.compile(r"\b([A-D])\b")


def extract_letter(text):
    m = CHOICE_RE.search(text or "")
    if m:
        return m.group(1).upper()
    tail = (text or "").strip()[-200:]          # fall back to a late standalone letter
    ms = LOOSE_RE.findall(tail)
    return ms[-1].upper() if ms else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval70.jsonl")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--adapter", default="", help="LoRA adapter dir; empty = base model control")
    ap.add_argument("--num-frames", type=int, default=100)
    ap.add_argument("--max-pixels", type=int, default=50176)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--fp32", action="store_true", default=False,
                    help="load in float32 instead of bfloat16")
    ap.add_argument("--thinking", action="store_true", default=False,
                    help="enable Qwen3.5 thinking mode (open <think> block)")
    ap.add_argument("--keep-think", action="store_true", default=False,
                    help="leave the template's empty <think></think> in the prompt "
                         "(GRPO's format); default strips it to match SFT/eval")
    ap.add_argument("--n-votes", type=int, default=1,
                    help=">1 samples N completions (one prefill, num_return_sequences=N) "
                         "and majority-votes the <choice> letters")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--answer-fallback", action="store_true", default=False,
                    help="on a TRUNCATED generation with no parsable <choice>, take a second turn "
                         "that hands the partial reasoning back and asks only for the letter, "
                         "constrained to A/B/C/D. Default OFF so previously reported numbers stay "
                         "reproducible; turn it on to measure its effect.")
    ap.add_argument("--tie-break", default="first", choices=["first", "C"],
                    help="on a vote tie: 'first' = highest-ranked sample, 'C' = majority class")
    a = ap.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=(torch.float32 if a.fp32 else torch.bfloat16),
        # --fp32 removes bf16 rounding from the comparison. Local and H100 produce identical
        # inputs and identical weights, yet answers differ on 10/70 with an 8-1 asymmetry
        # favouring local — too lopsided for symmetric rounding noise. If fp32 makes the two
        # machines agree, it was bf16 kernel differences; if local still wins, something else.
        device_map="cuda", attn_implementation="sdpa")
    tag = "BASE"
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        tag = f"LoRA({Path(a.adapter).name})"
    model.eval()

    # Vocab-pruned checkpoint support, so the SAME harness that produced our reported numbers can
    # also score the pruned model -- otherwise a pruned-vs-original comparison would confound the
    # prune with a change of harness. The tokenizer is not pruned, so it still emits original-space
    # ids that must be remapped before the embedding lookup (out-of-range = CUDA assert), and the
    # model generates in pruned space, so generated ids must be mapped back before decoding.
    # No vocab_remap.json => every line below is skipped and behaviour is byte-identical to before.
    RM = None
    _rmp = Path(MODEL) / "vocab_remap.json"
    if _rmp.exists():
        _r = json.load(open(_rmp))
        _extras = [int(x) for x in _r.get("extras", [])]
        RM = {"keep_low": _r["keep_low"], "added_start": _r["added_start"],
              "n_added": _r["n_added"], "n_new": _r["n_new"],
              "extra_pos": {t: _r["keep_low"] + k for k, t in enumerate(_extras)},
              "added_base": _r["keep_low"] + len(_extras),
              "fallback": {int(k): v for k, v in _r["fallback"].items()},
              "inv": (list(range(_r["keep_low"])) + _extras
                      + list(range(_r["added_start"], _r["added_start"] + _r["n_added"])))}
        assert len(RM["inv"]) == RM["n_new"], "vocab_remap.json inconsistent"
        tag += f" PRUNED({RM['n_new']})"
        print(f"vocab-pruned checkpoint: remapping into {RM['n_new']:,} rows", flush=True)

    def remap_batch(batch):
        """old-space batch -> pruned-space batch, realigning every per-token tensor."""
        src = batch["input_ids"][0].tolist()
        out, idx = [], []
        for i, t in enumerate(src):
            t = int(t)
            if t < RM["keep_low"]:
                nt = t
            elif t in RM["extra_pos"]:
                nt = RM["extra_pos"][t]
            elif RM["added_start"] <= t < RM["added_start"] + RM["n_added"]:
                nt = RM["added_base"] + (t - RM["added_start"])
            else:
                exp = RM["fallback"].get(t) or [0]
                out.extend(exp); idx.extend([i] * len(exp))
                continue
            out.append(nt); idx.append(i)
        if out == src:
            return batch, len(src)
        if max(out) >= RM["n_new"]:
            raise ValueError(f"remapped id {max(out)} >= vocab {RM['n_new']}")
        dev = batch["input_ids"].device
        nb = dict(batch)
        nb["input_ids"] = torch.tensor([out], dtype=torch.long, device=dev)
        # the byte fallback lengthens the sequence; M-RoPE needs mm_token_type_ids to match
        gi = torch.tensor(idx, dtype=torch.long, device=dev)
        for _k in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
            _t = batch.get(_k)
            if _t is not None and _t.shape[-1] == len(src):
                nb[_k] = _t[..., gi]
        return nb, len(out)

    recs = [json.loads(l) for l in open(a.data)]
    if a.limit:
        recs = recs[:a.limit]
    n = ok = parsed = n_fallback = 0
    out_rows = []
    for r in recs:
        man = Path(a.frames) / r["video_id"] / "frames.json"
        if not man.exists():
            continue
        paths = json.load(open(man))[:a.num_frames]
        query = ("Provide the detailed video description and answer to the question now: "
                 f"Question:{r['question']} {r['mcq_options']}")
        msgs = [{"role": "system", "content": [{"type": "text", "text": JUNIOR_SYSTEM}]},
                {"role": "user", "content": [{"type": "image"} for _ in paths] +
                                            [{"type": "text", "text": query}]}]
        # --thinking turns Qwen3.5 thinking mode ON (template emits an OPEN <think> for the
        # model to fill). Default is off: SFT and eval both strip the empty block, and the 2B
        # is known to loop when thinking is enabled. Test before betting an RL run on it.
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             **({"enable_thinking": True} if a.thinking else {}))
        # Qwen3.5's template appends an EMPTY <think></think> (non-thinking mode marker).
        # SFT and eval both strip it, so the model never sees it. GRPO does NOT strip it,
        # so a GRPO policy was optimised on the un-stripped format. --keep-think reproduces
        # that format here to test whether the mismatch costs anything.
        if not a.keep_think:
            text = re.sub(r"<think>\s*</think>\s*", "", text)
        images = [Image.open(p).convert("RGB") for p in paths]
        batch = processor(text=[text], images=images, return_tensors="pt",
                          images_kwargs={"max_pixels": a.max_pixels, "min_pixels": 3136}
                          ).to(model.device)
        plen = batch["input_ids"].shape[1]
        if RM is not None:
            batch, plen = remap_batch(batch)
        pkv = None
        with torch.no_grad():
            if a.n_votes > 1:
                gen = model.generate(**batch, max_new_tokens=a.max_new_tokens,
                                     do_sample=True, temperature=a.temperature, top_p=0.95,
                                     num_return_sequences=a.n_votes)
            elif a.answer_fallback:
                # keep the cache so the fallback turn can continue from it instead of re-prefilling
                # ~36k tokens of frames (see scripts/answer_fallback.py)
                _o = model.generate(**batch, max_new_tokens=a.max_new_tokens, do_sample=False,
                                    return_dict_in_generate=True, use_cache=True)
                gen, pkv = _o.sequences, _o.past_key_values
            else:
                gen = model.generate(**batch, max_new_tokens=a.max_new_tokens, do_sample=False)
        if RM is not None:                        # model generated in PRUNED space
            comps = [processor.tokenizer.decode(
                [RM["inv"][int(t)] for t in g[plen:] if 0 <= int(t) < RM["n_new"]],
                skip_special_tokens=True) for g in gen]
        else:
            comps = [processor.tokenizer.decode(g[plen:], skip_special_tokens=True) for g in gen]
        votes = [extract_letter(c) for c in comps]
        valid = [v for v in votes if v]
        if a.n_votes > 1 and valid:
            from collections import Counter
            cnt = Counter(valid).most_common()
            if len(cnt) == 1 or cnt[0][1] > cnt[1][1]:
                letter = cnt[0][0]
            else:                                   # tie
                letter = "C" if a.tie_break == "C" else valid[0]
        else:
            letter = valid[0] if valid else None
        comp = comps[0]

        # Truncation guard: the budget ran out mid-reasoning so no <answer> was ever emitted, and
        # the item would score wrong regardless of what the reasoning concluded. Hand the partial
        # reasoning back and ask only for the choice, constrained to A/B/C/D.
        used_fallback = False
        if a.answer_fallback and letter is None and \
                needs_fallback(comp, len(gen[0]) - plen, a.max_new_tokens):
            try:
                letter = fallback_choice(model, processor, msgs, images, comp,
                                         {"max_pixels": a.max_pixels, "min_pixels": 3136},
                                         prev_ids=gen[0].tolist(), past_key_values=pkv,
                                         remap=RM)
                used_fallback = letter is not None
                n_fallback += used_fallback
            except Exception as e:
                print(f"  fallback failed on idx={r['index']}: {e}", flush=True)

        n += 1
        parsed += letter is not None
        good = (letter == r["answer"])
        ok += good
        out_rows.append({"index": r["index"], "gt": r["answer"], "pred": letter,
                         "correct": bool(good), "votes": votes, "completion": comp,
                         "used_fallback": used_fallback})
        if n % 10 == 0:
            print(f"  {n}/{len(recs)} running acc={ok}/{n}={100*ok/n:.1f}%", flush=True)
    print(f"\n=== {tag}  max_pixels={a.max_pixels}  frames={a.num_frames} "
          f"model={MODEL} thinking={a.thinking} maxnew={a.max_new_tokens} ===")
    print(f"accuracy: {ok}/{n} = {100*ok/max(1,n):.1f}%   (parsed {parsed}/{n})"
          + (f"   [truncation-fallback recovered {n_fallback}]" if a.answer_fallback else ""))
    if a.out:
        json.dump(out_rows, open(a.out, "w"), indent=1)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
