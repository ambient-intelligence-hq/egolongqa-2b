"""Second-turn guard for thinking-truncation: never return an empty option.

The failure it fixes: with thinking enabled the model sometimes spends the whole token budget
reasoning and gets cut off mid-<video_description>, so it never emits <answer>. The item is then
scored wrong no matter what the reasoning had concluded -- and in the eval70 capture at 8192 tokens
this still happens (idx 286, 307 both ran the budget out). The old behaviour fell back to the
majority class "C", which throws away a completed line of reasoning.

The guard: hand the truncated reasoning back to the model in a new user turn and ask only for the
choice, with thinking off so it cannot run away a second time.

Two design decisions worth stating, because the obvious implementations are worse:

1. *Constrained, not prompted.* Rather than asking for a format and hoping, this teacher-forces the
   prefix "<answer><choice>" and takes argmax over exactly {A,B,C,D} -- single tokens 32-35, all
   below the vocab-prune cut so they survive pruning. One forward pass, no generation loop, and an
   invalid or empty answer is structurally impossible. A re-prompt could truncate again; this cannot.

2. *Full re-encode, deliberately.* Reusing the first pass's KV cache is the obvious optimisation
   and it does NOT work: M-RoPE derives cos/sin from the full attention length while the forwarded
   tail is short, so the rotary shapes disagree. Caught-and-ignored it is dead code; uncaught it
   makes the guard raise and fall through to the majority class -- the exact bug this prevents. The
   extra prefill is cheap beside the 8192-token generation that got us here, and only ~3% of items
   reach it.
"""
from __future__ import annotations

import re

FOLLOWUP = (
    "You ran out of generation budget before producing an answer, so your previous reply was cut "
    "off. Do not continue the description and do not reason further. Using only the reasoning you "
    "already produced above, give the final answer now in exactly this format and nothing else:\n"
    "<answer><choice>X</choice></answer>\n"
    "where X is exactly one of A, B, C, or D."
)

_CHOICE_RE = re.compile(r"<choice>\s*([A-D])\s*</choice>", re.I)
_PREFIX = "<answer><choice>"


def needs_fallback(text, gen_len, max_new_tokens):
    """True when the budget ran out AND no parsable choice was produced.

    Both conditions matter: a short reply with no <choice> is a different (format) failure that a
    second turn will not fix, and a truncated reply that still parsed needs no help.
    """
    return gen_len >= max_new_tokens and not _CHOICE_RE.search(text or "")


def fallback_choice(model, processor, base_msgs, images, partial_text, images_kwargs,
                    prev_ids=None, past_key_values=None, remap=None, logger=None):
    """Ask for the choice alone, constrained to A/B/C/D. Returns 'A'..'D', or None if it fails.

    remap: optional dict from a vocab-pruned checkpoint; the forced-prefix ids and the four letter
    ids must be mapped into pruned space, and the letters are looked up at their PRUNED positions.
    """
    import torch

    tok = processor.tokenizer
    msgs = list(base_msgs) + [
        {"role": "assistant", "content": [{"type": "text", "text": partial_text}]},
        {"role": "user", "content": [{"type": "text", "text": FOLLOWUP}]},
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text = re.sub(r"<think>\s*</think>\s*", "", text) + _PREFIX

    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCD"]

    def _map(ids):
        """-> (pruned-space ids, src_index). src_index[j] = position in `ids` that produced j."""
        if remap is None:
            return list(ids), list(range(len(ids)))
        out, src = [], []
        for i, t in enumerate(ids):
            t = int(t)
            if t < remap["keep_low"]:
                n = t
            elif t in remap["extra_pos"]:
                n = remap["extra_pos"][t]
            elif remap["added_start"] <= t < remap["added_start"] + remap["n_added"]:
                n = remap["added_base"] + (t - remap["added_start"])
            else:
                exp = remap["fallback"].get(t, [0])
                out.extend(exp); src.extend([i] * len(exp))
                continue
            out.append(n); src.append(i)
        return out, src

    full = processor(text=[text], images=images, return_tensors="pt",
                     images_kwargs=images_kwargs).to(model.device)
    full_ids, full_src = _map(full["input_ids"][0].tolist())

    # Re-encode rather than continuing from the first pass's KV cache. Cache reuse was tried and
    # does not work here: M-RoPE derives cos/sin from the FULL attention length while the forwarded
    # tail is short, so the rotary shapes disagree (94 vs 5694). Silently catching that would leave
    # dead code that never fires; one extra prefill is cheap next to the 8192-token generation that
    # got us here, and this path only runs on the ~3% of items that truncate.
    # prev_ids/past_key_values are accepted and ignored so callers need not know this.
    batch = dict(full)
    n_src = full["input_ids"].shape[-1]
    if remap is not None and full_ids != full["input_ids"][0].tolist():
        batch["input_ids"] = torch.tensor([full_ids], dtype=torch.long, device=model.device)
        gi = torch.tensor(full_src, dtype=torch.long, device=model.device)
        for _k in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
            _t = batch.get(_k)
            if _t is not None and _t.shape[-1] == n_src:
                batch[_k] = _t[..., gi]
    with torch.no_grad():
        logits = model(**batch).logits[0, -1]

    mapped = [_map([i])[0][0] for i in letter_ids]
    if any(i >= logits.shape[-1] for i in mapped):
        return None
    pick = int(torch.stack([logits[i] for i in mapped]).argmax())
    return "ABCD"[pick]
