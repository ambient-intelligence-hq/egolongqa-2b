#!/usr/bin/env python3
"""Merge the arm B LoRA into Qwen3.5-2B and save a standalone checkpoint for baking.

Why merge rather than serve base+adapter: vLLM's qwen3_5 declares SupportsLoRA, but multimodal
+ LoRA is finicky, and the submission contract rewards a single self-contained /models dir. A
merged checkpoint also makes the declared parameter count unambiguous — which matters because the
served model must be verified against the <=2B division, not assumed from the name.
"""
import argparse, json, torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

print(f"loading base {a.base} ...", flush=True)
m = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cpu")
n_base = sum(p.numel() for p in m.parameters())
print(f"  base params: {n_base:,} ({n_base/1e9:.3f} B)", flush=True)

print(f"applying adapter {a.adapter} ...", flush=True)
m = PeftModel.from_pretrained(m, a.adapter)
m = m.merge_and_unload()
n = sum(p.numel() for p in m.parameters())
print(f"  merged params: {n:,} ({n/1e9:.3f} B)  [unchanged by merging — LoRA folds in]", flush=True)

# report the split, since the <=2B division may or may not count the vision tower
vis = getattr(m, "visual", None) or getattr(getattr(m, "model", None), "visual", None)
if vis is not None:
    nv = sum(p.numel() for p in vis.parameters())
    print(f"  vision tower: {nv:,} ({nv/1e9:.3f} B)   language model: {(n-nv)/1e9:.3f} B", flush=True)

m.save_pretrained(a.out, safe_serialization=True)
AutoProcessor.from_pretrained(a.base).save_pretrained(a.out)
json.dump({"total_params": n, "base_params": n_base,
           "vision_params": (nv if vis is not None else None)},
          open(f"{a.out}/param_count.json", "w"), indent=1)
print(f"saved -> {a.out}", flush=True)
