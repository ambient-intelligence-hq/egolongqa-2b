#!/usr/bin/env python
"""LoRA SFT of Qwen3.5-2B as the EgoLongQA JUNIOR PERCEIVER (distillation from large teachers).

Task (matches ambient/tools/preempt_answer.py exactly, so train == inference):
  system(PREMTPT_FOR_ANSWER_TOOL_PROMPT) + user([~100 uniform frames] + question)
    -> <video_description>timestamped overview</video_description>
       <answer><choice>X</choice><reason>..</reason><citation>..</citation></answer>

Teacher targets are CORRECT junior passes harvested from the 122b/gemma/27b/35b agentic runs
(scripts/build_distill_dataset.py). Only the final assistant span is supervised.

Following the reference's insight that the *scored* decision should not be drowned out by the
long free-text span, the <choice> letter (what MCQ accuracy actually measures) is up-weighted
relative to the description/reason tokens.

LoRA r=16 / alpha=32 (same 1:2 ratio as the reference setup).

Usage:
  python src/train_lora.py --data data/train_junior.jsonl \
      --frames data/frames --out runs/armB
"""
import argparse, json, os, re, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

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

MODEL = "Qwen/Qwen3.5-2B"

# Up-weighting <choice> optimises the scored token directly, but it also pushes the model toward
# fitting the ANSWER rather than learning to perceive -- which is what hurts transfer to a test set
# whose answer distribution differs from val-700's 63% C-skew. Overridable via --choice-weight.
W_CHOICE_DEFAULT = 3.0
W_CHOICE = W_CHOICE_DEFAULT   # the <choice>X</choice> letter — what MCQ accuracy scores
W_TEXT   = 1.0   # description / reason / citation tokens
SEG_CHOICE, SEG_TEXT = 1, 2

processor = None
tok = None
IM_END = None
ASSIST_HDR = None


def init_processor(max_pixels):
    global processor, tok, IM_END, ASSIST_HDR
    processor = AutoProcessor.from_pretrained(MODEL)
    tok = processor.tokenizer
    IM_END = tok.convert_tokens_to_ids("<|im_end|>")
    ASSIST_HDR = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)


def build_messages(ex):
    """system + user([images] + query) + assistant(target). Images precede the query text,
    mirroring _build_messages / the junior tool's call shape."""
    user0 = [{"type": "image"} for _ in ex["images"]] + [{"type": "text", "text": ex["query"]}]
    return [{"role": "system", "content": [{"type": "text", "text": JUNIOR_SYSTEM}]},
            {"role": "user", "content": user0},
            {"role": "assistant", "content": [{"type": "text", "text": ex["target"]}]}]


def labels_and_weights(input_ids, target):
    """Supervise ONLY the final assistant span; up-weight the <choice> letter inside it."""
    ids = input_ids.tolist()
    labels = [-100] * len(ids)
    weights = [0.0] * len(ids)
    segments = [0] * len(ids)
    H = len(ASSIST_HDR)
    starts = [i for i in range(len(ids) - H + 1) if ids[i:i + H] == ASSIST_HDR]
    if not starts:
        return (torch.tensor(labels), torch.tensor(weights, dtype=torch.float32),
                torch.tensor(segments, dtype=torch.long))
    i = starts[-1] + H
    j = i
    while j < len(ids) and ids[j] != IM_END:
        j += 1
    end = min(j + 1, len(ids))                     # include closing <|im_end|>
    for k in range(i, end):
        labels[k] = ids[k]
        weights[k] = W_TEXT
        segments[k] = SEG_TEXT
    # Up-weight the <choice> letter. Re-tokenising a prefix does NOT align with the full
    # tokenisation (merges cross boundaries), so map char->token by walking the span's
    # decoded text cumulatively and marking whichever token covers the letter.
    m = re.search(r"<choice>\s*([A-D])\s*</choice>", target)
    if m:
        letter_char = m.start(1)
        pos = 0
        for k in range(i, end):
            piece = tok.decode([ids[k]])
            nxt = pos + len(piece)
            if pos <= letter_char < nxt:            # this token covers the answer letter
                weights[k] = W_CHOICE
                segments[k] = SEG_CHOICE
                break
            pos = nxt
    return (torch.tensor(labels), torch.tensor(weights, dtype=torch.float32),
            torch.tensor(segments, dtype=torch.long))


class Collator:
    def __init__(self, max_pixels, min_pixels=3136):
        self.max_pixels, self.min_pixels = max_pixels, min_pixels

    def __call__(self, examples):
        """Batched. The old collator took examples[0] only, which forced
        per_device_train_batch_size=1 and left the GPU badly underutilised.

        Right-padding is safe here because labels_and_weights() locates the supervised span by
        SEARCHING for the last assistant header and walking to the first <|im_end|> — it never
        assumes the target sits at the end of the sequence. Pad positions therefore keep
        label -100 / weight 0 and contribute nothing to the loss.
        """
        texts, images = [], []
        for ex in examples:
            t = processor.apply_chat_template(build_messages(ex), tokenize=False,
                                              add_generation_prompt=False)
            texts.append(re.sub(r"<think>\s*</think>\s*", "", t))
            images.extend(Image.open(p).convert("RGB") for p in ex["images"])
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True,
                          images_kwargs={"max_pixels": self.max_pixels,
                                         "min_pixels": self.min_pixels})
        B, L = batch["input_ids"].shape
        labels = torch.full((B, L), -100, dtype=torch.long)
        weights = torch.zeros((B, L), dtype=torch.float32)
        segments = torch.zeros((B, L), dtype=torch.long)
        for i, ex in enumerate(examples):
            lab, w, seg = labels_and_weights(batch["input_ids"][i], ex["target"])
            labels[i], weights[i], segments[i] = lab, w, seg
        batch["labels"] = labels
        batch["token_weights"] = weights
        batch["token_segments"] = segments
        return batch


class WeightedSFTTrainer(SFTTrainer):
    _SEG = {SEG_CHOICE: "choice", SEG_TEXT: "text"}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._seg_sum = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("token_weights")
        segments = inputs.pop("token_segments", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Gather ONLY the supervised positions, and never materialise a full-sequence copy.
        # Only the assistant span is supervised (~500 of ~10k positions — the 6.4k vision tokens
        # are all -100), but `logits[:, :-1, :].reshape(-1, V)` copies the whole non-contiguous
        # slice (47 GB at batch 8) before any masking, and `.float()` on that doubles it again.
        # Boolean-indexing the view gathers just the kept rows. cross_entropy already contributes
        # 0 for ignore_index, so this changes memory, not the value.
        lb = labels[:, 1:].to(logits.device)
        keep = lb != -100
        n = keep.sum().clamp(min=1)
        sl = logits[:, :-1, :][keep]                       # (n_kept, V), no full-tensor copy
        ce_kept = F.cross_entropy(sl.float(), lb[keep], reduction="none")
        sw = weights[:, 1:].to(logits.device)[keep].float()
        loss = (ce_kept * sw).sum() / n
        if segments is not None:
            seg = segments[:, 1:].to(ce_kept.device)[keep]   # `keep` is 2D — do not flatten first
            with torch.no_grad():
                for sid, name in self._SEG.items():
                    m = seg == sid
                    c = int(m.sum())
                    if c:
                        e = self._seg_sum.setdefault(name, [0.0, 0])
                        e[0] += float(ce_kept[m].sum()); e[1] += c
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *a, **k):
        if self._seg_sum:
            for name, (s, c) in self._seg_sum.items():
                if c:
                    logs[f"loss_{name}"] = s / c
            self._seg_sum = {}
        return super().log(logs, *a, **k)


def load_rows(data_path, frames_dir, num_frames):
    """One training row per teacher trace (augmentation); frames shared per video."""
    rows, missing = [], 0
    for line in open(data_path):
        r = json.loads(line)
        man = Path(frames_dir) / r["video_id"] / "frames.json"
        if not man.exists():
            missing += 1
            continue
        paths = json.load(open(man))[:num_frames]
        # frames.json may store paths relative to the tree it was written in. Re-root any
        # that do not resolve onto frames_dir/<video_id>/<file> so a manifest copied between
        # machines still works. ORDER is preserved -- the manifest is the contract, and the
        # trainer takes the first N verbatim.
        paths = [q if Path(q).exists()
                 else str(Path(frames_dir) / r["video_id"] / Path(q).name)
                 for q in paths]
        if len(paths) < num_frames * 0.5:
            missing += 1
            continue
        query = ("Provide the detailed video description and answer to the question now: "
                 f"Question:{r['question']} {r['mcq_options']}")
        for t in r["traces"]:
            rows.append({"images": paths, "query": query, "target": t["target"],
                         "index": r["index"], "video_id": r["video_id"]})
    return rows, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_junior.jsonl")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--out", default="runs/armB")
    ap.add_argument("--num-frames", type=int, default=100)
    ap.add_argument("--max-pixels", type=int, default=50176,
                    help="per-image pixel budget (50176 ~ 64 vision tokens/frame)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="per-device batch. EFFECTIVE batch = batch_size * grad_accum — keep that "
                         "product fixed when comparing runs, or the comparison is confounded")
    ap.add_argument("--attn", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)      # 1:2 ratio, same as reference
    ap.add_argument("--choice-weight", type=float, default=W_CHOICE_DEFAULT,
                    help="loss weight on the <choice> letter vs description/reason tokens "
                         "(3.0 = default; 1.0 = no up-weighting)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--report-to", default="none")
    a = ap.parse_args()

    global W_CHOICE
    W_CHOICE = a.choice_weight
    init_processor(a.max_pixels)
    print(f"choice-token loss weight: {W_CHOICE}  (text tokens = {W_TEXT})")
    rows, missing = load_rows(a.data, a.frames, a.num_frames)
    print(f"train rows: {len(rows)}  (skipped {missing} samples w/o materialized frames)")
    print(f"frames/sample={a.num_frames}  max_pixels={a.max_pixels}  "
          f"~{a.max_pixels // 784} vision tokens/frame  "
          f"~{a.num_frames * a.max_pixels // 784} vision tokens/sample")
    ds = Dataset.from_list(rows)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        # FA2 over SDPA: SDPA pads variable-length VL samples, so batching wasted compute and made
        # bs=1 look optimal. With FA2 throughput scales near-linearly with batch size.
        attn_implementation=a.attn)
    model.config.use_cache = False
    peft_cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(
        output_dir=a.out, per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=5,
        # one checkpoint per epoch (keep them all so each epoch can be evaluated separately)
        save_strategy="epoch", save_total_limit=None, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=a.report_to, run_name=os.path.basename(a.out),
        dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
        max_length=None)
    trainer = WeightedSFTTrainer(model=model, args=cfg, train_dataset=ds,
                                 data_collator=Collator(a.max_pixels), peft_config=peft_cfg)
    trainer.train(resume_from_checkpoint=a.resume or None)
    trainer.save_model(a.out)
    processor.save_pretrained(a.out)
    print("SAVED_ADAPTER", a.out)


if __name__ == "__main__":
    main()
