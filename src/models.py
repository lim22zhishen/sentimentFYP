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
import os
from functools import lru_cache

from src.config import HUGGINGFACE_TOKEN, get_device

# Multilingual sentiment with POSITIVE / NEUTRAL / NEGATIVE labels.
SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def asr_model_name() -> str:
    """Whisper model for faster-whisper (CTranslate2, no torchcodec).

    Set ``ASR_MODEL`` in ``.env`` (e.g. ``medium``, ``small``) to override. The
    default is ``large-v3`` on a GPU and ``small`` on CPU, where large-v3 is
    impractically slow.
    """
    return os.getenv("ASR_MODEL") or ("large-v3" if get_device() == "cuda" else "small")


@lru_cache(maxsize=1)
def load_asr_model():
    from faster_whisper import WhisperModel

    device = get_device()
    # float16 on GPU; int8 on CPU, where float16 is unsupported or slow.
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(asr_model_name(), device=device, compute_type=compute_type)


@lru_cache(maxsize=1)
def load_sentiment_pipeline():
    from transformers import pipeline as hf_pipeline

    return hf_pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        # transformers wants an int device: 0 = first GPU, -1 = CPU.
        device=0 if get_device() == "cuda" else -1,
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
    pipe.to(torch.device(get_device()))
    return pipe
