"""
NLLB-200 translation engine using HuggingFace transformers.

Model: facebook/nllb-200-distilled-1.3B  (FP32, ~2.6 GB)
Input:  source text  +  source language tag (e.g. "kat_Geor")
Output: dict with 'translation' (str)
"""

from __future__ import annotations

import threading
from typing import Dict

_MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
_DEFAULT_TARGET = "eng_Latn"
_MAX_INPUT_CHARS = 400  # stay well within NLLB's 512-token limit

# ── ISO 639-1 (Whisper) -> NLLB BCP-47 style tag ──────────────────────
ISO_TO_NLLB: Dict[str, str] = {
    "ka": "kat_Geor",   # Georgian
    "ru": "rus_Cyrl",   # Russian
    "zh": "zho_Hans",   # Chinese (Simplified)
    "ar": "arb_Arab",   # Arabic
    "ko": "kor_Hang",   # Korean
    "ja": "jpn_Jpan",   # Japanese
    "tr": "tur_Latn",   # Turkish
    "es": "spa_Latn",   # Spanish
    "fr": "fra_Latn",   # French
    "de": "deu_Latn",   # German
    "pt": "por_Latn",   # Portuguese
    "en": "eng_Latn",   # English
    "uk": "ukr_Cyrl",   # Ukrainian
    "hi": "hin_Deva",   # Hindi
    "it": "ita_Latn",   # Italian
    "pl": "pol_Latn",   # Polish
    "nl": "nld_Latn",   # Dutch
    "vi": "vie_Latn",   # Vietnamese
    "th": "tha_Thai",   # Thai
    "he": "heb_Hebr",   # Hebrew
    "sv": "swe_Latn",   # Swedish
    "cs": "ces_Latn",   # Czech
    "ro": "ron_Latn",   # Romanian
    "da": "dan_Latn",   # Danish
    "fi": "fin_Latn",   # Finnish
    "hu": "hun_Latn",   # Hungarian
    "el": "ell_Grek",   # Greek
    "id": "ind_Latn",   # Indonesian
    "ms": "zsm_Latn",   # Malay
    "bn": "ben_Beng",   # Bengali
    "ta": "tam_Taml",   # Tamil
    "te": "tel_Telu",   # Telugu
    "ur": "urd_Arab",   # Urdu
    "fa": "pes_Arab",   # Persian
    "az": "azj_Latn",   # Azerbaijani
    "hy": "hye_Armn",   # Armenian
    "my": "mya_Mymr",   # Burmese
    "km": "khm_Khmr",   # Khmer
}

# ── Lazy singleton ─────────────────────────────────────────────────────
_lock = threading.Lock()
_pipeline = None  # transformers.pipeline instance


def _get_pipeline():
    """Return the NLLB translation pipeline, loading the model exactly once."""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                import torch
                from transformers import (
                    AutoModelForSeq2SeqLM,
                    AutoTokenizer,
                    pipeline,
                )

                tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    _MODEL_NAME,
                    torch_dtype=torch.float32,
                )
                _pipeline = pipeline(
                    "translation",
                    model=model,
                    tokenizer=tokenizer,
                    device="cpu",
                )
    return _pipeline


def translate(
    text: str,
    source_lang: str,
    target_lang: str = _DEFAULT_TARGET,
) -> dict:
    """
    Translate *text* from *source_lang* to *target_lang*.

    Parameters
    ----------
    text : str
        Source text to translate.
    source_lang : str
        NLLB language tag (e.g. "kat_Geor") **or** ISO 639-1 code ("ka").
        ISO codes are mapped automatically.
    target_lang : str
        NLLB target tag.  Defaults to "eng_Latn".

    Returns
    -------
    dict  {"translation": str}
    """
    # Resolve ISO code to NLLB tag if needed.
    if len(source_lang) <= 3 and "_" not in source_lang:
        source_lang = ISO_TO_NLLB.get(source_lang, source_lang)

    # If source is already the target language, pass through unchanged.
    if source_lang == target_lang:
        return {"translation": text}

    # Truncate to stay within token budget.
    if len(text) > _MAX_INPUT_CHARS:
        text = text[:_MAX_INPUT_CHARS]

    pipe = _get_pipeline()
    result = pipe(
        text,
        src_lang=source_lang,
        tgt_lang=target_lang,
        max_length=512,
    )
    translation = result[0]["translation_text"] if result else ""
    return {"translation": translation}


def warmup() -> None:
    """Force-load the model by translating a short test string."""
    translate("test", source_lang="kat_Geor", target_lang="eng_Latn")
