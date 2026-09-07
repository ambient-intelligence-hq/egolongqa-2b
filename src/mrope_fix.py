"""Make Qwen3.5 M-RoPE robust to a prompt whose leading image got truncated.

TRL's GRPO loss forward can receive `input_ids` whose FIRST image-pad group is partial
(the prompt was cut mid-image) while `image_grid_thw` still describes every full image.
Stock `get_rope_index` sizes each vision block from the grid, so llm_positions ends up
longer than the attended tokens -> "shape mismatch ... cannot be broadcast".

Fix: walk the modality groups exactly like the original, but size each vision block by the
ACTUAL number of pad tokens in that group. Full images are unchanged (group == grid); only a
truncated group is shortened, and its positions are taken from the TAIL of the full block so
the surviving pads keep their true spatial positions.
"""
import itertools
import torch


def install(image_pad_id):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    cls = next(getattr(M, n) for n in dir(M)
               if isinstance(getattr(M, n), type) and hasattr(getattr(M, n), "get_rope_index"))
    orig = cls.get_rope_index
    stats = {"fixed": 0}

    def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None,
                       attention_mask=None, mm_token_type_ids=None, **kw):
        if image_grid_thw is None or input_ids is None or mm_token_type_ids is None:
            return orig(self, input_ids, image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw, attention_mask=attention_mask,
                        mm_token_type_ids=mm_token_type_ids, **kw)
        merge = getattr(self.config.vision_config, "spatial_merge_size", 2)
        # fast path: if every image group already matches its grid, use the original
        need_fix = False
        gi = iter(image_grid_thw)
        try:
            for b in range(input_ids.shape[0]):
                itt = mm_token_type_ids[b]
                if attention_mask is not None:
                    itt = itt[attention_mask[b].bool()]
                for k, g in itertools.groupby(itt.tolist()):
                    if k != 0:
                        n = len(list(g))
                        gt = next(gi)
                        if n != int(gt[0]) * int(gt[1]) * int(gt[2]) // (merge * merge):
                            need_fix = True
                            raise StopIteration
                    else:
                        list(g)
        except StopIteration:
            pass
        if not need_fix:
            return orig(self, input_ids, image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw, attention_mask=attention_mask,
                        mm_token_type_ids=mm_token_type_ids, **kw)

        stats["fixed"] += 1
        if stats["fixed"] == 1:
            print("[mrope-fix] truncated leading image detected; sizing vision blocks "
                  "from actual pad counts", flush=True)
        B, L = input_ids.shape
        position_ids = torch.ones(3, B, L, dtype=torch.long, device=input_ids.device)
        deltas = []
        gi = iter(image_grid_thw)
        for b in range(B):
            itt = mm_token_type_ids[b]
            mask_b = attention_mask[b].bool() if attention_mask is not None else None
            if mask_b is not None:
                itt = itt[mask_b]
            blocks, cur = [], 0
            for k, g in itertools.groupby(itt.tolist()):
                n = len(list(g))
                if k == 0:                                   # text
                    blocks.append(torch.arange(n, device=input_ids.device)
                                  .view(1, -1).expand(3, -1) + cur)
                    cur += n
                else:                                        # image
                    gt = next(gi)
                    full = self.get_vision_position_ids(cur, gt, 1, merge,
                                                        device=input_ids.device)
                    if full.shape[1] != n:                   # truncated group -> keep the tail
                        full = full[:, -n:] if n <= full.shape[1] else \
                            torch.cat([full, full[:, -1:].expand(3, n - full.shape[1])], dim=1)
                    blocks.append(full)
                    cur += max(int(gt[1]), int(gt[2])) // merge
            llm = torch.cat(blocks, dim=1).reshape(3, -1)
            if mask_b is not None:
                position_ids[:, b, mask_b] = llm.to(position_ids.device)
            else:
                position_ids[:, b] = llm.to(position_ids.device)
            deltas.append(llm.max() + 1 - int(itt.shape[0]))
        return position_ids, torch.tensor(deltas, device=input_ids.device).unsqueeze(1)

    cls.get_rope_index = get_rope_index
    print(f"[mrope-fix] installed on {cls.__name__}", flush=True)
