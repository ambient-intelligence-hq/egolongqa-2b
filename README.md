# Distilling Long-Video perception into a Sub-2B Model, EgoLongQA ≤2B — winning entry

<!-- TODO: replace the arXiv id below once the tech report is posted -->
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Collection-Wearable%20AI%20ECCV%202026-FFD21E)](https://huggingface.co/collections/ambient-intelligence-labs/wearables-ai-workshop-eccv-2026)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-egolongqa--2b--distill--adapter-FFD21E)](https://huggingface.co/infinitylogesh/egolongqa-2b-distill-adapter)
[![arXiv](https://img.shields.io/badge/arXiv-2609.07154-B31B1B.svg)](https://arxiv.org/abs/2609.07154)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-1st%20%E2%89%A42B-1F883D)](https://huggingface.co/spaces/facebook/wearable-ai-leaderboard)

🏆 First place, **≤2B division**, EgoLongQA - Meta Wearable AI Grand Challenge (ECCV 2026)
([leaderboard](https://huggingface.co/spaces/facebook/wearable-ai-leaderboard)) — **0.8279**
on the held-out test set, at **1.9985 B** parameters.

A single Qwen3.5-2B answers a four-option question about a ~10-minute egocentric video in one
greedy forward pass. No retrieval, no tool calls, no ensemble.

It is built in three steps:

1. **Distil** the perception step of an agentic pipeline into the 2B student, from teacher passes
   filtered to those that answered correctly.
2. **Prune** the vocabulary from 248,320 to 143,469 rows to get under the 2 B limit, with provably
   identical logits on retained rows.
3. **Serve** at 100 frames and `max_pixels=331776`.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

One 80 GB GPU is enough for every step. Training the adapter takes ~4 h; the rest is minutes.

## What you need to supply

| | |
|---|---|
| Videos | EgoLongQA val split (`facebook/wearable-ai`, config `egolongqa`) |
| Frames | 100 uniform JPEGs per video, long side 768 — see below |
| Traces | teacher outputs in the junior format (`data/train_junior.jsonl`) |

Our traces and evaluation splits are published at
`ambient-intelligence-labs/egolongqa-synth-annotations` and `ambient-intelligence-labs/egolongqa-junior-distill`
. `src/build_distill_dataset.py` regenerates them from raw agentic runs.

**Frame extraction is part of the contract, not a preprocessing detail.** Use exactly:

```bash
ffmpeg -hide_banner -loglevel error -y -i VIDEO \
  -vf "fps=min(2,100/DURATION),scale=768:768:force_original_aspect_ratio=decrease" \
  -q:v 3  frames/%03d.jpg      # then take the first 100
```

The trainer and evaluator both read `frames.json` and take the first N — **the manifest is the
contract.** If it lists more frames than the teacher actually saw, the student learns to describe
footage it was never shown.

Paths inside `frames.json` may be relative to whatever tree wrote them. Both scripts re-root any
path that does not resolve onto `--frames/<video_id>/<file>`, so a manifest copied between machines
still works; the order is always preserved.

---

## Reproduce

### 1. Shuffle the options (do not skip)

```bash
python src/shuffle_options.py --in data/train_junior.jsonl --out data/train_junior_shuf.jsonl
```

The validation gold is **63.4 % C** (A=8, B=205, C=444, D=43), so "always C" scores 62.9 %. Training
on unshuffled options teaches that prior instead of the task: on option-permuted evaluation a
standard distilled model drops **25.7 points** while the shuffled one is stable.

### 2. Train the adapter — one epoch

```bash
python src/train_lora.py \
  --data data/train_junior_shuf.jsonl --frames data/frames \
  --out runs/armB --epochs 1 --lora-r 16 --lora-alpha 32 \
  --choice-weight 3.0 --max-pixels 50176
```

Three settings are load-bearing:

- **One epoch.** Epoch 2 lost 4.3 points on the debiased metric in every run — it re-absorbs the
  prior that shuffling removed.
- **`--max-pixels 50176` for training, 331776 for inference.** Training cheap and inferring high is
  worth **+5.8 points** and cuts training cost ~6×. The adapter learns the task, not the resolution.
- **`--choice-weight 3.0`.** Dropping it to 1.0 was worse on every evaluation.

### 3. Merge and prune to ≤2 B

```bash
python src/merge_lora.py --base Qwen/Qwen3.5-2B \
  --adapter runs/armB/checkpoint-367 --out work/merged

python src/vocab_pruning/prune_vocab.py \
  work/merged work/pruned 143000 extra_ids_egolongqa_v2.json
chmod -R a+rX work/pruned            # prune writes -rw-------; Docker bakes ZERO-BYTE weights otherwise

python src/vocab_pruning/count_params.py work/pruned   # expect 1,998,506,816
```

Expected: vocab 248,320 → 143,469, **1.9985 B**, `model.safetensors` md5
`69337fedd6cef84cd54f5266a785f2e5`.

> **The output side is not lossless.** Input-side pruning has a byte-level fallback; the output side
> has none. If the model's argmax lands on a removed row it emits a *different word* — a semantic
> change, not a spelling one. The keep-set must therefore be captured on **the GPU that will serve
> the model**, because the same checkpoint emits different completions on different hardware:
>
> ```bash
> python src/vocab_pruning/capture_output_vocab.py --model work/merged --data data/eval70.jsonl
> python src/vocab_pruning/build_extras.py --cap 1165 --out extra_ids.json
> python src/vocab_pruning/verify.py --orig work/merged --pruned work/pruned
> ```
>
> `verify.py` checks coverage, byte round-trip, argmax safety, logit equivalence on retained rows
> (expect `|diff| = 0.00e+00`) and greedy-generation identity.

### 4. Evaluate — always on both metrics

```bash
# skewed (benchmark option order)
python src/evaluate.py --model work/pruned --data data/eval70.jsonl \
  --frames data/frames --num-frames 100 --max-pixels 331776 --max-new-tokens 8192

# debiased twin -- same videos and questions, options permuted to uniform gold
python src/evaluate.py --model work/pruned --data data/eval70_shuf.jsonl \
  --frames data/frames --num-frames 100 --max-pixels 331776 --max-new-tokens 8192
```

`--model` takes a base, merged or pruned checkpoint (it also honours `$STUDENT_MODEL`).
Add `--adapter runs/armB/checkpoint-367` to evaluate an unmerged adapter instead.

Expected on 70 held-out videos: **65.7 % skewed / 74.3 % debiased**.

> Report both. `eval70` inherits the 63 % C-skew and *cannot distinguish* a model that perceives from
> one that has absorbed the prior. `eval70_shuf` is the same videos and questions with options
> permuted to uniform gold.

## Acknowledgments

We thank the organizers of the [Wearable AI Workshop @ ECCV 2026](https://wearable-ai-workshop.github.io/) for the benchmark, the dataset, and the evaluation infrastructure , and for their responsiveness throughout the challenge.

---
## Citation

```bibtex
@misc{ambient2026egolongqa,
  title  = {Ambient @ EgoLongQA 2026: Distilling Long-Video perception into a Sub-2B Model},
  author = {Umapathi, Logesh Kumar},
  year   = {2026},
  note   = {AI Wearables Challenge 2026, EgoLongQA {$\leq$}2B division --- 1st place}
}
```
