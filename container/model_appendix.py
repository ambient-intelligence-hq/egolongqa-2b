

# ===========================================================================
# Participant extension: distilled Qwen3.6-27B junior perceiver (EgoLongQA).
# Registered below under the built-in keys so any --model-type the runtime
# passes on the HF backend routes here.
# ===========================================================================

_JUNIOR_SYSTEM = """You are video analysis expert, You are given an entire egocentric (first-person) video, followed by a multiple-choice question with lettered options (A, B, C, D). Inspect ALL of the provided visual content across the whole video before answering — the\x20
answer may depend on events anywhere in the timeline.

First create a detailed description of the video (not less than 50 words) and then answer the question given to you based on the description ( with evidence from time stamps):

Your description will be used to provide initial context to a video research agent.

Your description should cover the following aspects:
- The overall context of the video
- How the video starts, middle and ends
- Notable locations, events, actions and persons in the video
- Any other relevant information
- You should provide the description grouped by timestamp to give an understand of the video


Provide the description in the following format (plain text, NO JSON):
<video_description>
{description}
</video_description>

<answer>
<choice>C</choice>
<reason>one or two sentences justifying the chosen option, referencing what you saw</reason>
<citation> list of start and end timestamps in seconds </citation>
</answer>

Rules for the answer block:
- <choice> must contain ONLY a single letter: A, B, C, or D. Nothing else.
- <reason> is one or two short sentences. Do not put JSON, lists, or timestamps guesses inside these tags.
- <citation> should be a list of start and end timestamps in seconds.

Things to avoid:
- You will only have details to the range of timestamps and not exact timestamps of each frame. So you should not
try to guess the timestamp of an action or even within the range.\x20
For example, if you are given a range of frames between **246.5 seconds - 493.0 seconds:** , you should not mention in your description that the even occurred at 289th sec. As frames are sampled broadly, your guess could be wrong.
- Provide the answer accurately as possible given the information you have.
"""

import re as _re_junior

# ---------------------------------------------------------------------------
# FRAME EXTRACTION — override the starter kit's cv2 sampler.
#
# This is not an optimisation, it is a correctness fix. Every number we report was measured on
# frames produced by ambient's VideoFrameTools (ffmpeg, max_frame_dimention=768). The starter kit
# decodes the video itself with cv2 and samples DIFFERENTLY in three ways, all verified against the
# real files rather than inferred:
#
#   1. TEMPORAL. cv2 takes frame indices start + i*step, i.e. t = i*D/N (interval STARTS).
#      ffmpeg's fps filter rounds to nearest and lands on interval MIDPOINTS, t = (i+0.5)*D/N.
#      Measured on 02597f9f4c0ec8cc (603 s, 15 fps): our frame 0 best-matches container index 44,
#      an offset of +2.93 s -- exactly the half-step (0.5 * 603/100 = 3.02 s). Every frame is
#      ~3 s off, in a benchmark whose questions turn on when things happen.
#   2. SPATIAL. Ours are scaled to a 768 long side BEFORE the processor sees them
#      (1440x1080 -> 768x576); cv2 hands over native resolution and lets max_pixels downscale.
#      Same cap, different resampling chain, different pixels.
#   3. PHOTOMETRIC. Ours are JPEG q:v 3; cv2's are raw decoded frames.
#
# So we shell out to the SAME ffmpeg command, rather than reimplementing it:
#     -vf fps=min(2, N/duration),scale=768:768:force_original_aspect_ratio=decrease  -q:v 3
# Falling back to the starter implementation if anything goes wrong -- serving slightly-off frames
# beats serving none.
_starter_extract_frames = extract_frames
_OVERVIEW_FPS = 2                 # ambient VideoFrameTools.OVERVIEW_FPS
_MAX_FRAME_DIM = 768              # materialize_distill_frames.py passes max_frame_dimention=768
_FFMPEG_QSCALE = 3                # VideoFrameTools.FFMPEG_QSCALE


def _probe_duration(video_path):
    import subprocess
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         video_path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def extract_frames(video_path, intervals=None, frames_per_interval=4, max_frames=32):
    """Frames identical to the ones every reported number was measured on. See note above."""
    import glob
    import shutil
    import subprocess
    import tempfile
    from PIL import Image

    if intervals is not None:                 # not the longqa path; leave it to the starter code
        return _starter_extract_frames(video_path, intervals, frames_per_interval, max_frames)
    tmp = None
    try:
        duration = _probe_duration(video_path)
        if not (duration > 0):
            raise ValueError(f"bad duration {duration!r}")
        eff = float(_OVERVIEW_FPS)
        if max_frames:
            eff = min(eff, max_frames / duration)
        if eff <= 0:
            eff = float(_OVERVIEW_FPS)
        # TMPDIR is /dev/shm in this image: the validator runs a --read-only rootfs
        tmp = tempfile.mkdtemp(prefix="junior_frames_")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video_path,
             "-vf", f"fps={eff},scale={_MAX_FRAME_DIM}:{_MAX_FRAME_DIM}:"
                    "force_original_aspect_ratio=decrease",
             "-q:v", str(_FFMPEG_QSCALE),
             os.path.join(tmp, "frame_%09d.jpg")],
            check=True, capture_output=True, text=True)
        paths = sorted(glob.glob(os.path.join(tmp, "frame_*.jpg")))
        if not paths:
            raise RuntimeError("ffmpeg produced no frames")
        # training read manifest[:N] -- take the FIRST max_frames, never a re-subsample
        frames = [Image.open(p).convert("RGB") for p in paths[:max_frames]]
        for f in frames:
            f.load()
        logger.info("extract_frames(ffmpeg): %s -> %d frames @ %.4f fps",
                    os.path.basename(video_path), len(frames), eff)
        return frames
    except Exception:
        logger.exception("ffmpeg frame extraction failed; falling back to starter cv2 sampler")
        return _starter_extract_frames(video_path, intervals, frames_per_interval, max_frames)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


_CHOICE_RE = _re_junior.compile(r"<choice>\s*([A-D])\s*</choice>", _re_junior.I)
_LOOSE_RE = _re_junior.compile(r"\b([A-D])\b")
# Pull the question + options back out of the runner's LONGQA_PROMPT_TEMPLATE so
# we can re-issue them in the exact shape the junior was distilled on.
_Q_RE = _re_junior.compile(r"Question:\s*(.*?)\s*(?:\n\nOptions:\s*(.*))?$", _re_junior.S)


class Junior27BModel(VideoQAModel):
    """Distilled Qwen3.6-27B junior perceiver (single greedy pass, ~100 frames).

    Reshapes the runner's generic MCQ prompt into the junior's trained contract
    (system + '<video_description>...<answer><choice>X</choice>...'), generates
    with a large token budget regardless of the caller's max_new_tokens (the
    junior emits a long description before the answer), then returns the parsed
    letter (with a majority-class 'C' fallback on a parse failure).
    """

    def __init__(self, model_id: str = "/models/junior27b") -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading junior-27B: %s ...", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.max_pixels = int(os.environ.get("JUNIOR_MAX_PIXELS", "331776"))
        self.min_answer_tokens = int(os.environ.get("JUNIOR_MAX_NEW_TOKENS", "2048"))
        # Qwen3.5 thinking mode. Set by the image so the two candidate builds differ ONLY in this
        # default and nothing else -- on the pinned stack, eval70_shuf gave 75.7% with thinking and
        # 71.4% without, a 3-item gap that is not significant at n=70, so both are submitted and
        # compared rather than one being chosen on a coin-flip.
        self.thinking = os.environ.get("JUNIOR_THINKING", "0") == "1"

        # Vocab-pruned checkpoint (<=2B division): the tokenizer is deliberately NOT pruned, so it
        # still emits ids in the original 248K space while the embedding has ~143K rows. Feeding
        # those ids straight in runs off the end of the table -- a CUDA device-side assert, which
        # kills the worker and scores the whole shard empty. Absent the file, this is a no-op and
        # the unpruned path is byte-for-byte unchanged.
        self.remap = None
        _rm = os.path.join(model_id, "vocab_remap.json")
        if os.path.exists(_rm):
            import json as _json
            with open(_rm) as _fh:
                _r = _json.load(_fh)
            self.keep_low = _r["keep_low"]
            self.added_start = _r["added_start"]
            self.n_added = _r["n_added"]
            self.n_new = _r["n_new"]
            self.extras = [int(x) for x in _r.get("extras", [])]
            self.extra_pos = {t: self.keep_low + k for k, t in enumerate(self.extras)}
            self.added_base = self.keep_low + len(self.extras)
            self.fallback = {int(k): v for k, v in _r["fallback"].items()}
            # new_id -> old_id: the model GENERATES in pruned space, and the unpruned tokenizer
            # would decode those ids as completely different text.
            self.inv = (list(range(self.keep_low)) + self.extras
                        + list(range(self.added_start, self.added_start + self.n_added)))
            # NOT an assert. This runs at model load on the organisers' held-out set: an assert
            # here would abort __init__, kill the worker, and score the ENTIRE shard empty over a
            # consistency property that is verified at BUILD time anyway. Log it and continue --
            # any id that would actually run off the embedding is caught per-item in _generate,
            # which degrades one answer to the majority class instead of losing every answer.
            # (Asserts are also stripped under `python -O`, so they are not a dependable check.)
            if len(self.inv) != self.n_new:
                logger.error("vocab_remap.json inconsistent: inv=%d n_new=%d — continuing with "
                             "per-item range guard", len(self.inv), self.n_new)
            self.remap = True
            logger.info("Vocab-pruned checkpoint: remapping %d -> %d ids.",
                        self.added_start + self.n_added, self.n_new)

        # Log the ffmpeg build once at load. The version affects which frames get sampled (its
        # ffprobe duration feeds eff_fps), so having it in the organisers' logs makes an unexpected
        # binary diagnosable after the fact. We ship the base image's 4.4.2 on measured grounds --
        # see the Containerfile note. Purely informational: it must never raise, because a version
        # surprise costs a few borderline frames whereas failing here loses the whole shard.
        try:
            import subprocess as _sp
            _v = _sp.run(["ffmpeg", "-version"], capture_output=True, text=True,
                         timeout=30).stdout.splitlines()[0]
            logger.info("frame extractor: %s", _v)
        except Exception:
            logger.exception("could not determine ffmpeg version (extraction may fall back to cv2)")
        logger.info("Junior-27B loaded (max_pixels=%d, pruned=%s).",
                    self.max_pixels, bool(self.remap))

    def _remap_ids(self, ids):
        """old-space -> (new-space ids, src_index). src_index[j] = position in `ids` that made j."""
        out, src = [], []
        for i, t in enumerate(ids):
            t = int(t)
            if t < self.keep_low:
                n = t                                    # identity: the common case
            elif t in self.extra_pos:
                n = self.extra_pos[t]
            elif self.added_start <= t < self.added_start + self.n_added:
                n = self.added_base + (t - self.added_start)
            else:
                exp = self.fallback.get(t) or [0]        # byte-exact expansion; content preserved
                out.extend(exp)
                src.extend([i] * len(exp))
                continue
            out.append(n)
            src.append(i)
        return out, src

    @staticmethod
    def _user_text(messages: list) -> str:
        for m in messages:
            if m.get("role") == "user":
                return str(m.get("content", ""))
        return str(messages[-1].get("content", "")) if messages else ""

    def _build_query(self, user_text: str) -> str:
        m = _Q_RE.search(user_text)
        if m:
            q = (m.group(1) or "").strip()
            opts = (m.group(2) or "").strip()
            # LONGQA_PROMPT_TEMPLATE puts a trailing paragraph after the options:
            #   "Answer with ONLY the single letter of the correct option (A, B, C, or D).
            #    Do not include any other text."
            # `(.*)` under re.S swallows it, and it then CONTRADICTS our system prompt, which asks
            # for <video_description> followed by <answer>. Measured: with it attached the model
            # emits a bare letter and skips the description entirely -- items that take the full
            # 8192-token budget in our reference finished in seconds in-container. Keep only the
            # first paragraph (the options are a single line); splitting on the blank line survives
            # a reworded instruction, whereas matching its text would not.
            opts = opts.split("\n\n")[0].strip()
            body = f"Question:{q} {opts}".strip()
        else:  # fall back to the raw prompt text if the template changed
            body = user_text.strip()
        return "Provide the detailed video description and answer to the question now: " + body

    @staticmethod
    def _parse_letter(text: str):
        m = _CHOICE_RE.search(text or "")
        if m:
            return m.group(1).upper()
        tail = (text or "").strip()[-200:]
        ms = _LOOSE_RE.findall(tail)
        return ms[-1].upper() if ms else None

    def generate(self, frames, messages, max_new_tokens: int = 16) -> str:
        # One malformed item must never kill the worker: an uncaught exception here takes down
        # the process and every remaining question in the shard scores empty, turning a single
        # bad sample into a zero. Set JUNIOR_STRICT=1 in development to fail loud instead.
        try:
            return self._generate(frames, messages, max_new_tokens)
        except Exception:
            if os.environ.get("JUNIOR_STRICT"):
                raise
            logger.exception("junior generate() failed; returning majority-class fallback")
            return "C"

    def _generate(self, frames, messages, max_new_tokens: int = 16) -> str:
        import torch

        query = self._build_query(self._user_text(messages))
        mm = [
            {"role": "system", "content": [{"type": "text", "text": _JUNIOR_SYSTEM}]},
            {"role": "user", "content": [{"type": "image"} for _ in frames]
                                        + [{"type": "text", "text": query}]},
        ]
        text = self.processor.apply_chat_template(
            mm, tokenize=False, add_generation_prompt=True,
            **({"enable_thinking": True} if self.thinking else {})
        )
        if not self.thinking:
            # non-thinking mode: strip the template's EMPTY <think></think> marker, exactly as SFT
            # and every eval that produced our reported numbers do
            text = _re_junior.sub(r"<think>\s*</think>\s*", "", text)
        batch = self.processor(
            text=[text], images=list(frames) if frames else None,
            return_tensors="pt",
            images_kwargs={"max_pixels": self.max_pixels, "min_pixels": 3136},
        ).to(self.model.device)
        budget = max(int(max_new_tokens), self.min_answer_tokens)

        if self.remap:
            src_ids = batch["input_ids"][0].tolist()
            dst, src_index = self._remap_ids(src_ids)
            if dst != src_ids:
                dev = batch["input_ids"].device
                batch = dict(batch)
                batch["input_ids"] = torch.tensor([dst], dtype=torch.long, device=dev)
                # The byte fallback makes the sequence LONGER, and Qwen3.5 derives M-RoPE
                # positions from a per-token mm_token_type_ids that must stay the same length as
                # input_ids -- otherwise vision features splice at the wrong positions. Re-gather
                # every per-token tensor through src_index so all of them stay aligned.
                idx = torch.tensor(src_index, dtype=torch.long, device=dev)
                for _k in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
                    _t = batch.get(_k)
                    if _t is not None and _t.shape[-1] == len(src_ids):
                        batch[_k] = _t[..., idx]
            if dst and max(dst) >= self.n_new:           # must never happen; fail loud
                raise ValueError(f"remapped id {max(dst)} >= pruned vocab {self.n_new}")
            plen = len(dst)
        else:
            plen = batch["input_ids"].shape[1]

        with torch.no_grad():
            res = self.model.generate(**batch, max_new_tokens=budget, do_sample=False,
                                      return_dict_in_generate=True, use_cache=True)
        seq = res.sequences[0].tolist()
        raw = seq[plen:]
        new_ids = [self.inv[int(t)] for t in raw if 0 <= int(t) < self.n_new] if self.remap else raw
        gen = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        letter = self._parse_letter(gen)

        # Truncation guard. With thinking on, the model sometimes spends the whole budget reasoning
        # and is cut off before emitting <answer> -- the item then scores wrong regardless of what
        # the reasoning concluded, and returning the majority class throws that reasoning away.
        # Hand the partial reasoning back and ask ONLY for the letter, with the answer constrained
        # to A/B/C/D so a second truncation cannot produce an empty option.
        if letter is None and len(raw) >= budget:
            try:
                letter = self._forced_choice(mm, frames, gen, seq,
                                             getattr(res, "past_key_values", None))
            except Exception:
                logger.exception("truncation fallback failed; using majority class")
        return letter or "C"   # majority-class fallback (GT is 63% C)

    _FOLLOWUP = (
        "You ran out of generation budget before producing an answer, so your previous reply was "
        "cut off. Do not continue the description and do not reason further. Using only the "
        "reasoning you already produced above, give the final answer now in exactly this format "
        "and nothing else:\n<answer><choice>X</choice></answer>\n"
        "where X is exactly one of A, B, C, or D."
    )

    def _forced_choice(self, base_msgs, frames, partial, prev_ids, past_key_values):
        """Second turn, constrained to A/B/C/D. Returns a letter, or None.

        Teacher-forces the prefix '<answer><choice>' and takes argmax over exactly the four letter
        tokens, so the result is always a valid option -- a re-prompt could simply truncate again.

        This re-encodes rather than continuing from the first pass's KV cache. Reusing the cache
        looked like the obvious optimisation and was tried: it fails, because M-RoPE derives cos/sin
        from the FULL attention length while the forwarded tail is short, so the rotary embedding
        shapes disagree (94 vs 5694). Caught and silently re-encoded it would be dead code that
        never fires; left uncaught it makes the guard raise and fall through to the majority class,
        which is exactly the bug this guard exists to prevent. One extra prefill is cheap next to
        the 8192-token generation that got us here, and it only fires on the ~3% truncated items.
        `past_key_values` is accepted and ignored so callers need not know this.
        """
        import torch

        tok = self.processor.tokenizer
        msgs = list(base_msgs) + [
            {"role": "assistant", "content": [{"type": "text", "text": partial}]},
            {"role": "user", "content": [{"type": "text", "text": self._FOLLOWUP}]},
        ]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = _re_junior.sub(r"<think>\s*</think>\s*", "", text) + "<answer><choice>"
        full = self.processor(
            text=[text], images=list(frames) if frames else None, return_tensors="pt",
            images_kwargs={"max_pixels": self.max_pixels, "min_pixels": 3136},
        ).to(self.model.device)

        src = full["input_ids"][0].tolist()
        ids, idx = self._remap_ids(src) if self.remap else (src, list(range(len(src))))
        dev = self.model.device
        b = dict(full)
        if self.remap and ids != src:
            b["input_ids"] = torch.tensor([ids], dtype=torch.long, device=dev)
            # the byte fallback lengthens the sequence; realign every per-token tensor so M-RoPE
            # and the vision splice stay aligned with pixel_values
            gi = torch.tensor(idx, dtype=torch.long, device=dev)
            for _k in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
                _t = b.get(_k)
                if _t is not None and _t.shape[-1] == len(src):
                    b[_k] = _t[..., gi]
        with torch.no_grad():
            logits = self.model(**b).logits[0, -1]

        letters = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCD"]
        if self.remap:
            letters = [self._remap_ids([i])[0][0] for i in letters]
        if any(i >= logits.shape[-1] for i in letters):
            return None
        return "ABCD"[int(torch.stack([logits[i] for i in letters]).argmax())]


# Route every built-in key to the junior on the HF backend, and default the
# weights path to the baked merged model so `--llm-model` is optional.
# JUNIOR_MODEL_DIR lets the same appendix serve the 27B large-track image and the pruned-2B
# small-track image without a code fork -- the image sets it, nothing else changes.
_JUNIOR_DIR = os.environ.get("JUNIOR_MODEL_DIR", "/models/junior27b")
for _k in ("junior27b", "qwen", "llama4"):
    MODEL_REGISTRY[_k] = Junior27BModel
    DEFAULT_MODEL_IDS[_k] = _JUNIOR_DIR
    DEFAULT_GPU_COUNTS[_k] = 1
    DEFAULT_TP_SIZES[_k] = 1
    DEFAULT_BATCH_SIZES[_k] = 1
