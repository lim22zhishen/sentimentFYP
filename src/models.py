"""Cached local model loaders.

Each loader is wrapped in ``st.cache_resource`` so the model is downloaded and
moved to the device only once per session. Models download from the HuggingFace
Hub on first use and are then cached on disk (``~/.cache/huggingface``).
"""

import inspect

import streamlit as st
import torch
from faster_whisper import WhisperModel
from transformers import pipeline as hf_pipeline
from pyannote.audio import Pipeline as DiarizationPipeline

from src.config import ASR_COMPUTE_TYPE, DEVICE, HF_DEVICE, HUGGINGFACE_TOKEN

# Multilingual Whisper (faster-whisper / CTranslate2). Uses the GPU and avoids
# torchcodec, which fails to load on Windows.
ASR_MODEL = "large-v3"
# Multilingual sentiment with POSITIVE / NEUTRAL / NEGATIVE labels.
SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


@st.cache_resource(show_spinner="Loading speech-to-text model…")
def load_asr_model():
    return WhisperModel(ASR_MODEL, device=DEVICE, compute_type=ASR_COMPUTE_TYPE)


@st.cache_resource(show_spinner="Loading sentiment model…")
def load_sentiment_pipeline():
    return hf_pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        device=HF_DEVICE,
    )


@st.cache_resource(show_spinner="Loading speaker-diarization model…")
def load_diarization_pipeline():
    if not HUGGINGFACE_TOKEN:
        st.error(
            "A HuggingFace token is required for speaker diarization. Add "
            "`HUGGINGFACE_TOKEN` to your `.env`, and accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
        st.stop()

    # The auth keyword changed across pyannote versions (3.x: use_auth_token,
    # 4.x: token); pick whichever the installed version accepts.
    params = inspect.signature(DiarizationPipeline.from_pretrained).parameters
    auth_kwarg = "token" if "token" in params else "use_auth_token"
    pipe = DiarizationPipeline.from_pretrained(
        DIARIZATION_MODEL, **{auth_kwarg: HUGGINGFACE_TOKEN}
    )
    pipe.to(torch.device(DEVICE))
    return pipe
