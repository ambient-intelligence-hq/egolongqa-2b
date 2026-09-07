#!/usr/bin/env python3
"""Permute MCQ option order in a junior-perceiver training file so no letter prior is learnable.

WHY: val-700 gold is A=8 / B=205 / C=444 / D=43 -- an always-C floor of 63.4%. The distilled
students reproduce that prior almost exactly (C-rate 61-63%), and `eval70` is drawn from the same
700 with the same skew, so it cannot detect the problem. An unknown share of the score may be
"guess C when unsure" rather than perception, and the hidden test set may not share the skew.

Reuses the permutation approach of scripts/balance_annotation_letters.py, adapted to the training
schema: as well as `mcq_options` and `answer`, every teacher trace's `<choice>X</choice>` must be
rewritten to the new letter, or the target would contradict the prompt.

Usage:
  python src/shuffle_options.py --in data/train_junior.jsonl \
      --out data/train_junior_shuf.jsonl
"""
import argparse, collections, json, random, re

LETTERS = "ABCD"
OPT_SPLIT = re.compile(r"(?=\b[A-D][\.\)]\s)")
PREFIX = re.compile(r"^\s*([A-D])[\.\)]\s*")
CHOICE = re.compile(r"(<choice>\s*)([A-D])(\s*</choice>)")


def split_options(s):
    parts = [p.strip() for p in OPT_SPLIT.split(s) if p.strip()]
    return parts if len(parts) == 4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    # deal target letters round-robin over a shuffled order -> exactly uniform, not just unbiased
    order = list(range(len(rows)))
    rng.shuffle(order)
    target_of = {ri: LETTERS[i % 4] for i, ri in enumerate(order)}

    changed = skipped = traces_fixed = 0
    for ri, r in enumerate(rows):
        gold = str(r.get("answer", "")).upper()[:1]
        opts = split_options(r.get("mcq_options", ""))
        if not opts or gold not in LETTERS:
            skipped += 1
            continue
        bodies = [PREFIX.sub("", o).strip() for o in opts]
        gi = LETTERS.index(gold)
        gold_body = bodies[gi]
        others = [b for i, b in enumerate(bodies) if i != gi]
        rng.shuffle(others)

        tgt = target_of[ri]
        new, oi = [], 0
        for L in LETTERS:
            if L == tgt:
                new.append(f"{L}. {gold_body}")
            else:
                new.append(f"{L}. {others[oi]}"); oi += 1
        r["mcq_options"] = " ".join(new)
        r["answer"] = tgt
        # the trace targets assert the old letter; rewrite or the supervision contradicts the prompt
        for t in r.get("traces", []):
            t["target"], n = CHOICE.subn(lambda m: m.group(1) + tgt + m.group(3), t["target"])
            traces_fixed += n
        changed += 1

    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    dist = collections.Counter(r["answer"] for r in rows)
    print(f"wrote {len(rows)} rows -> {a.out}")
    print(f"  permuted {changed}, skipped {skipped}, trace <choice> rewrites {traces_fixed}")
    print(f"  gold distribution: {dict(sorted(dist.items()))}")
    tot = sum(dist.values())
    print(f"  max share {100*max(dist.values())/tot:.1f}%  (uniform = 25%)")


if __name__ == "__main__":
    main()
