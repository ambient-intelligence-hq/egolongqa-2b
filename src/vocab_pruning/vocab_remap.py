"""Serving-side id remapping for the vocab-pruned junior.

The tokenizer is deliberately NOT pruned, so it keeps emitting ids in the original 248K space
while the embedding matrix has only ~143K rows. Every id must therefore be remapped before it
reaches the model, or the lookup runs off the end of the table -- a CUDA device-side assert, which
kills the worker and scores the entire shard empty. That is the single most dangerous failure mode
of this whole change, so `remap` is exhaustive by construction and `assert_in_range` is cheap.

Two things here that EgoProactive's serving path did not need:

1. `mm_token_type_ids` realignment. A pruned token byte-expands into several, which makes the
   sequence longer -- and Qwen3.5 derives M-RoPE positions from a per-token mm_token_type_ids that
   must stay the same length as input_ids. `remap` therefore also returns `src_index`, the original
   position each new position came from, so any per-token tensor can be re-gathered exactly.
   (Vision specials live in the added block and are always kept, so they never expand; only text
   positions can grow.)

2. `inverse`. EgoProactive only read a yes/no logit pair and never decoded. Our junior GENERATES,
   and it generates ids in the PRUNED space, which the unpruned tokenizer would decode as entirely
   different text. Generated ids must be mapped back before decoding.
"""
from __future__ import annotations

import json
import os


class VocabRemap:
    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "vocab_remap.json")) as fh:
            r = json.load(fh)
        self.keep_low: int = r["keep_low"]
        self.added_start: int = r["added_start"]
        self.n_added: int = r["n_added"]
        self.n_new: int = r["n_new"]
        self.extras: list[int] = [int(x) for x in r.get("extras", [])]
        self.extra_pos: dict[int, int] = {t: self.keep_low + k for k, t in enumerate(self.extras)}
        self.added_base: int = self.keep_low + len(self.extras)
        self.fallback: dict[int, list[int]] = {int(k): v for k, v in r["fallback"].items()}
        # new_id -> old_id, for decoding what the model generates
        self.inv: list[int] = (list(range(self.keep_low)) + self.extras
                               + list(range(self.added_start, self.added_start + self.n_added)))
        assert len(self.inv) == self.n_new, "remap table does not match declared vocab size"

    def new_of(self, t: int):
        """Forward map for a single id, or None if the row was pruned."""
        if t < self.keep_low:
            return t                                    # identity: the overwhelmingly common case
        if t in self.extra_pos:
            return self.extra_pos[t]
        if self.added_start <= t < self.added_start + self.n_added:
            return self.added_base + (t - self.added_start)
        return None

    def remap(self, ids):
        """old-space ids -> (new-space ids, src_index).

        src_index[j] is the position in `ids` that produced output position j, so a per-token
        tensor T aligned with `ids` is realigned as [T[i] for i in src_index].
        """
        out: list[int] = []
        src: list[int] = []
        for i, t in enumerate(ids):
            n = self.new_of(t)
            if n is None:
                exp = self.fallback.get(t)
                if not exp:                             # unreachable: fallback is exhaustive
                    exp = [0]
                out.extend(exp)
                src.extend([i] * len(exp))
            else:
                out.append(n)
                src.append(i)
        return out, src

    def inverse(self, ids):
        """new-space ids (what the model generated) -> old-space ids, for the unpruned tokenizer."""
        n = self.n_new
        return [self.inv[int(t)] for t in ids if 0 <= int(t) < n]

    def assert_in_range(self, ids) -> None:
        if ids:
            hi = max(int(t) for t in ids)
            if hi >= self.n_new:
                raise ValueError(f"remapped id {hi} >= pruned vocab {self.n_new}")


def apply_to_batch(rm: VocabRemap, batch):
    """Remap a processor batch in place-ish; returns a new dict safe to pass to the model.

    Realigns every per-token tensor (attention_mask, mm_token_type_ids, token_type_ids) through
    src_index so lengths and M-RoPE stay consistent when the byte fallback grows the sequence.
    """
    import torch

    src_ids = batch["input_ids"][0].tolist()
    dst, src_index = rm.remap(src_ids)
    rm.assert_in_range(dst)
    if dst == src_ids:
        return batch                                     # nothing pruned: byte-identical fast path

    dev = batch["input_ids"].device
    out = dict(batch)
    out["input_ids"] = torch.tensor([dst], dtype=torch.long, device=dev)
    idx = torch.tensor(src_index, dtype=torch.long, device=dev)
    for key in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
        t = batch.get(key)
        if t is not None and t.shape[-1] == len(src_ids):
            out[key] = t[..., idx]
    return out
