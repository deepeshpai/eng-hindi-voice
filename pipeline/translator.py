"""
Stage 4 — English → Hindi Translation (fully local, no API key).

This module is responsible for one thing: running the MT model.
Pre-processing (abbreviation expansion, time expressions) lives in
`pipeline.preprocessing`; named-entity protection lives in
`pipeline.entity_guard`; tone adjustment lives here via `apply_tone`.

Backend
-------
NLLB-200 — facebook/nllb-200-distilled-600M (~1.2 GB)
    Best quality; handles slang, technical text, and mixed register.
    Supports entity-guard round-trip (opaque placeholder tokens survive).
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from config import TranslationConfig
from pipeline.entity_guard import protect, restore
from pipeline.preprocessing import preprocess
from utils import get_logger

log = get_logger(__name__)

# ── Tone tables ───────────────────────────────────────────────────────────────

_CASUAL_TO_FORMAL: list[tuple[str, str]] = [
    ("तू ", "आप "), ("तुम ", "आप "), ("तुम्हें", "आपको"), ("तुम्हारा", "आपका"),
    ("तुम्हारी", "आपकी"), ("तुम्हारे", "आपके"), ("तुमसे", "आपसे"),
    ("तुझे", "आपको"), ("तेरा", "आपका"), ("तेरी", "आपकी"), ("तेरे", "आपके"),
]
_FORMAL_TO_CASUAL: list[tuple[str, str]] = [(b, a) for a, b in _CASUAL_TO_FORMAL]


def apply_tone(hindi: str, tone: str) -> str:
    """Substitute Hindi pronouns to match the requested register."""
    if tone == "formal":
        for casual, formal in _CASUAL_TO_FORMAL:
            hindi = hindi.replace(casual, formal)
    elif tone == "casual":
        for formal, casual in _FORMAL_TO_CASUAL:
            hindi = hindi.replace(formal, casual)
    return hindi


# ── Translator ────────────────────────────────────────────────────────────────

class Translator:
    """
    Translates English → Hindi using NLLB-200 locally.

    Thread-safe: inference runs in a dedicated single-thread executor so it
    never blocks the asyncio event loop.
    """

    def __init__(self, cfg: TranslationConfig) -> None:
        self._cfg = cfg
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="translate")
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None

    # ── Initialization ────────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        device = self._cfg.device
        if device != "auto":
            return device
        try:
            import torch  # type: ignore
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def initialize(self) -> None:
        """Load NLLB-200 (blocking). Call once before serving requests."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        mid = self._cfg.NLLB_MODEL_ID
        log.info("Loading NLLB '%s' (first run downloads ~1.2 GB)…", mid)
        self._tokenizer = AutoTokenizer.from_pretrained(mid)
        device = self._resolve_device()
        self._model = AutoModelForSeq2SeqLM.from_pretrained(mid).to(device)
        log.info("NLLB ready on %s.", device)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_nllb(self, text: str) -> str:
        import torch  # type: ignore

        tok = self._tokenizer
        model = self._model
        device = next(model.parameters()).device  # type: ignore

        tok.src_lang = "eng_Latn"  # type: ignore
        inputs = tok(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)  # type: ignore

        tgt_id = tok.convert_tokens_to_ids("hin_Deva")  # type: ignore
        with torch.no_grad():
            out = model.generate(  # type: ignore
                **inputs,
                forced_bos_token_id=tgt_id,
                max_length=512,
                num_beams=4,
                early_stopping=True,
            )
        return tok.decode(out[0], skip_special_tokens=True)  # type: ignore

    def _translate_sync(self, english: str, tone: str) -> str:
        effective_tone = tone if self._cfg.tone == "auto" else self._cfg.tone
        guarded, entity_map = protect(english)
        preprocessed = preprocess(guarded)
        hindi = self._run_nllb(preprocessed)
        hindi = restore(hindi, entity_map)
        return apply_tone(hindi.strip(), effective_tone)

    # ── Public API ────────────────────────────────────────────────────────────

    async def translate(self, english: str, tone: str = "auto") -> str:
        """Async wrapper — runs inference in the thread-pool executor."""
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()
        result = await loop.run_in_executor(
            self._executor, self._translate_sync, english, tone
        )
        log.debug(
            "NLLB %.0f ms: %r → %r",
            (time.perf_counter() - t0) * 1000,
            english[:40],
            result[:40],
        )
        return result

    def close(self) -> None:
        self._executor.shutdown(wait=False)
