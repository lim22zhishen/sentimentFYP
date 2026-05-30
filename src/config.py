"""Configuration: secrets and compute device.

This app runs entirely on local models (no OpenAI). The only secret needed is a
HuggingFace token, used to download the gated pyannote diarization model.

Secrets are resolved as: OS environment / ``.env`` first, then ``st.secrets``
(so the same code works locally and on Streamlit Cloud, where the token may be
stored under the legacy key ``token``).
"""

import os

# Copy model files instead of symlinking them in the HuggingFace cache. Avoids
# the Windows "required privilege is not held" (WinError 1314) symlink error on
# machines without Developer Mode / admin rights. Set before HF libs import.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import streamlit as st
import torch
from dotenv import load_dotenv

# Load variables from a local .env file if present (no-op on Streamlit Cloud).
load_dotenv()


def get_secret(env_name, secrets_key=None, default=None):
    """Return a secret from env vars, then st.secrets, else ``default``."""
    value = os.getenv(env_name)
    if value:
        return value

    try:
        if secrets_key is not None and secrets_key in st.secrets:
            return st.secrets[secrets_key]
        if env_name in st.secrets:
            return st.secrets[env_name]
    except Exception:
        pass

    return default


# Needed only for speaker diarization (gated pyannote model). May be None; the
# diarization loader reports a friendly error if it's missing when actually used.
HUGGINGFACE_TOKEN = get_secret("HUGGINGFACE_TOKEN", "token")

# Compute device. Uses the GPU when available (e.g. an NVIDIA CUDA card).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# transformers `pipeline(device=...)` wants an int: 0 = first GPU, -1 = CPU.
HF_DEVICE = 0 if DEVICE == "cuda" else -1
# faster-whisper compute type: float16 on GPU, int8 on CPU.
ASR_COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
