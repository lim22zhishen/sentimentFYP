"""Cached local model loaders.

Framework-agnostic core: loaders are memoized with ``functools.lru_cache`` (so a
model is downloaded and moved to the device only once per process) and raise on
error instead of touching Streamlit. Models download from the HuggingFace Hub on
first use and are then cached on disk (``~/.cache/huggingface``).

The heavy third-party imports (faster-whisper, transformers, pyannote, torch)
are deferred into the loader bodies so importing this module stays cheap and
test-friendly.
"""

import inspect
from functools import lru_cache

from src.config import ASR_COMPUTE_TYPE, DEVICE, HF_DEVICE, HUGGINGFACE_TOKEN

# Multilingual Whisper (faster-whisper / CTranslate2). Uses the GPU and avoids
# torchcodec, which fails to load on Windows.
ASR_MODEL = "large-v3"
# Multilingual sentiment with POSITIVE / NEUTRAL / NEGATIVE labels.
SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


@lru_cache(maxsize=1)
def load_asr_model():
    from faster_whisper import WhisperModel

    return WhisperModel(ASR_MODEL, device=DEVICE, compute_type=ASR_COMPUTE_TYPE)


@lru_cache(maxsize=1)
def load_sentiment_pipeline():
    from transformers import pipeline as hf_pipeline

    return hf_pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        device=HF_DEVICE,
    )


@lru_cache(maxsize=1)
def load_diarization_pipeline():
    import torch
    from pyannote.audio import Pipeline as DiarizationPipeline

    if not HUGGINGFACE_TOKEN:
        raise RuntimeError(
            "A HuggingFace token is required for speaker diarization. Add "
            "HUGGINGFACE_TOKEN to your .env, and accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    # The auth keyword changed across pyannote versions (3.x: use_auth_token,
    # 4.x: token); pick whichever the installed version accepts.
    params = inspect.signature(DiarizationPipeline.from_pretrained).parameters
    auth_kwarg = "token" if "token" in params else "use_auth_token"
    pipe = DiarizationPipeline.from_pretrained(
        DIARIZATION_MODEL, **{auth_kwarg: HUGGINGFACE_TOKEN}
    )
    pipe.to(torch.device(DEVICE))
    return pipe
