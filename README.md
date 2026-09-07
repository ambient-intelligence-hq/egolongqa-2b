# EgoLongQA ≤2B — winning entry

First place, **≤2B division**, EgoLongQA track of the AI Wearables Challenge 2026
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
`infinitylogesh/egolongqa-synth-annotations` and `infinitylogesh/egolongqa-junior-distill`
(private; request access). `src/build_distill_dataset.py` regenerates them from raw agentic runs.

**Frame extraction is part of the contract, not a preprocessing detail.** Use exactly:

```bash
ffmpeg -hide_banner -loglevel error -y -i VIDEO \
  -vf "fps=min(2,100/DURATION),scale=768:768:force_original_aspect_ratio=decrease" \
  -q:v 3  frames/%03d.jpg      # then take the first 100
```

The trainer and evaluator both read `frames.json` verbatim and take the first N — **the manifest is
the contract.** If your manifest lists more frames than the teacher actually saw, the student learns
to describe footage it was never shown.

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
python src/evaluate.py --model work/pruned --data data/eval70.jsonl \
  --frames data/frames --num-frames 100 --max-pixels 331776 --max-new-tokens 8192

python src/evaluate.py --model work/pruned --data data/eval70_shuf.jsonl ...   # debiased twin
```

**`--max-new-tokens 8192` is not optional.** The default of 768 truncates 3–9 % of completions
mid-description so they never emit an answer and score wrong regardless of what the model would have
concluded. Every number we produced before fixing this was wrong by about one question.

Expected on 70 held-out videos: **65.7 % skewed / 74.3 % debiased**.

> Report both. `eval70` inherits the 63 % C-skew and *cannot distinguish* a model that perceives from
> one that has absorbed the prior. `eval70_shuf` is the same videos and questions with options
> permuted to uniform gold.

### 5. Build the container

`container/Containerfile.extension` and `container/model_appendix.py` go into the organisers'
starter kit. See the comments in both — every one marks a bug that cost real time.

---

## Things that will bite you

- **Do consistency checks at build time, never at model load.** An `assert` in `__init__` kills the
  worker and scores the *entire shard* empty. Asserts are also stripped under `python -O`.
- **The starter kit's frame sampler is not this one.** It decodes with cv2 at interval *starts*;
  ffmpeg's `fps` filter lands on *midpoints*. That is a ~3 s offset on every frame, on a benchmark
  about *when* things happen. `container/model_appendix.py` overrides it.
- **Keep ffmpeg 4.4.2** (the base image default). A 6.1.2 static build scored worse on both modes
  (65/70 vs 68/70 and 70/70). Do not "upgrade" it.
- **Verify per-sample, not on aggregate.** All four container bugs we hit preserved total accuracy
  while changing individual answers. The acceptance gate is per-sample agreement plus an explained
  diff list.
- **A container run returning `{'C': 70}`** is not a 25.7 % score — it means every generation failed
  into the fallback, usually because another job shared the GPU. Check the letter distribution.
- **~3–5 items at n=70 sit on the decision boundary.** A 0.02 MAE input change (invisible) flipped 3
  answers. Quote ±3 items of frame jitter alongside any container number.

## Layout

```
src/train_lora.py            LoRA SFT on teacher traces
src/evaluate.py              greedy eval, skewed and debiased
src/shuffle_options.py       option permutation + gold relabelling
src/merge_lora.py            adapter -> full checkpoint
src/build_distill_dataset.py agentic run logs -> training rows
src/mrope_fix.py             Qwen3.5 M-RoPE fix for truncated leading images
src/answer_fallback.py       second turn when thinking exhausts the budget
src/vocab_pruning/           profile -> build keep-set -> prune -> verify
container/                   model appendix + Containerfile extension
prompts/junior_system.txt    the system prompt (md5-guarded, 2000 bytes)
```

## Citation

```bibtex
@misc{ambient2026egolongqa,
  title  = {Ambient @ EgoLongQA 2026: Distilling Perception, Not Orchestration,
            into a Sub-2B Model},
  author = {Umapathi, Logesh Kumar},
  year   = {2026},
  note   = {AI Wearables Challenge 2026, EgoLongQA {$\leq$}2B division --- 1st place}
}
```
