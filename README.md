# Sentiment Analysis App

A [Streamlit](https://streamlit.io/) app that analyzes sentiment from **text** or
**audio** — running **entirely on local models** (no OpenAI / external API). For
audio it transcribes speech (Whisper), detects the language (and can translate
it to English), identifies who spoke when (pyannote diarization), and charts how
sentiment changes over time. It uses your **GPU** automatically when one is available.

## Features

- **Text mode** – paste a conversation, one turn per line, optionally prefixed
  with the speaker (`Alice: hello`); each line is labelled
  POSITIVE / NEUTRAL / NEGATIVE with a confidence score and plotted turn by turn.
- **Audio mode** – upload `wav / mp3 / ogg / m4a / flac`:
  - Transcription with Whisper (`large-v3` on GPU, `small` on CPU — configurable),
    with voice-activity detection so silence and music aren't transcribed
  - Language detection, plus optional English translation (a tick box — it runs
    Whisper a second time, so it roughly doubles transcription time)
  - Speaker diarization (`pyannote/speaker-diarization-3.1`); you can give the
    number of speakers if you know it, which improves accuracy
  - Word-level speaker alignment: when one Whisper segment covers two speakers
    (e.g. a question and its quick answer), it is split between them
  - Per-speaker, multilingual sentiment and a timeline chart
- **Chart & summary** – each point is the sentiment *polarity*,
  P(positive) − P(negative), from −1 to +1, so a confident and a borderline
  prediction look different; the per-speaker summary averages it.
- **Exports** – CSV / JSON of the results table, plus a timestamped transcript
  for audio.

## Models (downloaded on first run)

All models are pulled from the HuggingFace Hub the first time you use them and
cached on disk (`~/.cache/huggingface`, a few GB total):

| Task | Model | Library |
|---|---|---|
| Speech-to-text / translation | Whisper `large-v3` (GPU) / `small` (CPU) | faster-whisper (CTranslate2) |
| Sentiment (multilingual) | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | transformers |
| Speaker diarization (gated) | `pyannote/speaker-diarization-3.1` | pyannote.audio |

> On Windows, transcription uses **faster-whisper** rather than the transformers
> ASR pipeline, because the latter routes audio through `torchcodec`, which
> doesn't load on Windows. Diarization feeds pyannote an in-memory waveform for
> the same reason.

To use a different Whisper size, set `ASR_MODEL` in `.env` (e.g.
`ASR_MODEL=medium`). Any faster-whisper model name works: `tiny`, `base`,
`small`, `medium`, `large-v3`, `turbo`, …

## Project structure

```
sentimentFYP/
├── app.py                  # Streamlit UI / entry point (compute + render)
├── src/
│   ├── config.py           # token loading + device (CPU/GPU) selection
│   ├── models.py           # cached loaders for the 3 local models
│   ├── schemas.py          # dataclasses passed between pipeline and UI
│   ├── sentiment.py        # local sentiment + conversation splitting
│   ├── audio.py            # transcription, translation, diarization, ffmpeg
│   ├── evaluation.py       # accuracy / F1 / confusion-matrix metrics
│   └── ui_helpers.py       # tables, charts, speaker summary, exports
├── scripts/
│   ├── evaluate_sentiment.py   # score the sentiment model on a labelled CSV
│   └── sample_sentiment.csv    # example of the CSV format
├── tests/                  # fast tests (no GPU / no downloads)
├── .github/workflows/      # CI: runs the tests on every push
├── .streamlit/config.toml
├── .env.example            # template for the HuggingFace token + options
├── requirements.txt
├── requirements-dev.txt    # test deps (pytest)
└── README.md
```

## Run locally

> ⚠️ **Use Python 3.10 or 3.11.** The diarization/torch stack is not reliable on
> Python 3.12+ on Windows yet — stick to 3.11.

All steps install into a project-local virtual environment (`.venv/`) so your
global Python stays clean.

### 1. Create & activate a virtual environment

```powershell
# From the project root. If your default `python` is 3.12+, point at a 3.11
# interpreter explicitly, e.g. `py -3.11 -m venv .venv`.
python -m venv .venv

# Activate — Windows PowerShell:
.\.venv\Scripts\Activate.ps1
#   (execution-policy error? run once: Set-ExecutionPolicy -Scope Process RemoteSigned)
# Windows cmd:        .\.venv\Scripts\activate.bat
# macOS / Linux:      source .venv/bin/activate
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- **GPU (NVIDIA):** `requirements.txt` pulls CUDA 12.8 torch builds (required for
  RTX 50-series / Blackwell GPUs; also fine on older CUDA cards). This is a large
  download (~2–3 GB).
- **CPU-only:** remove the `--extra-index-url` line at the top of
  `requirements.txt` and install plain `torch`/`torchaudio` instead. Everything
  still works, just slower for transcription.
- **ffmpeg** is bundled via `imageio-ffmpeg` — no manual install needed.

Verify the GPU is picked up:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### 3. Add your HuggingFace token

```powershell
copy .env.example .env      # PowerShell / cmd   (cp on macOS/Linux)
```

Put your token in `.env` as `HUGGINGFACE_TOKEN=...`
(create one at https://huggingface.co/settings/tokens), and **accept the gated
model terms once** at
https://huggingface.co/pyannote/speaker-diarization-3.1.

> The token is only needed for diarization (audio mode). Text-mode sentiment
> works without it.

### 4. Run

```powershell
streamlit run app.py
```

Opens at http://localhost:8501. The first audio run downloads the models, so it
will take a while; subsequent runs are fast.

## First run & model sizes

Models download from the HuggingFace Hub the first time each is used and are
cached under `~/.cache/huggingface` (Windows: `C:\Users\<you>\.cache\huggingface`).
Plan for a few GB of download and disk on first use:

| Model | Approx. download | When it loads |
|---|---|---|
| Whisper `large-v3` (faster-whisper, GPU default) | ~1.5 GB | first transcription |
| Whisper `small` (faster-whisper, CPU default) | ~0.5 GB | first transcription |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | ~1.1 GB | first sentiment scoring |
| `pyannote/speaker-diarization-3.1` (+ segmentation/embedding) | ~0.5 GB | first diarization |

What to expect:
- **First audio analysis is slow** — it downloads all three models *and* warms
  them up on the GPU. Later runs reuse the already-loaded models (each loader
  is cached for the life of the Streamlit process with `functools.lru_cache`),
  so they're much faster.
- **Text-only mode** needs just the sentiment model (no HF token required).
- VRAM: the defaults target an ~8 GB GPU. On a smaller card, see Troubleshooting.

## Tests

Pure logic (conversation splitting, label normalization, polarity, speaker
assignment, audio preprocessing, transcript building, evaluation metrics) and
the text and audio app flows (driven with Streamlit's `AppTest`, models
stubbed) are covered by fast tests that need no GPU or model downloads:

```powershell
pip install -r requirements-dev.txt
pytest
```

The diarization tests need PyTorch; without it they're skipped and everything
else still runs. GitHub Actions runs the full suite on every push
(`.github/workflows/tests.yml`).

## Evaluating the sentiment model

For the report, you can measure how well the sentiment model agrees with
human labels. Prepare a CSV with a `text` column and a `label` column
(`negative` / `neutral` / `positive`, or `neg` / `neu` / `pos`), then run:

```powershell
python scripts/evaluate_sentiment.py my_labelled_data.csv
# other column names, and save the misclassified rows for error analysis:
python scripts/evaluate_sentiment.py data.csv --text-col sentence --label-col sentiment --errors misclassified.csv
```

It prints accuracy, macro F1, per-label precision / recall / F1 and a confusion
matrix. `scripts/sample_sentiment.csv` shows the format; at 15 rows it is only
an example. For meaningful numbers, use a few hundred labelled lines, ideally
from the kind of conversations the app is meant for.

## Troubleshooting

- **GPU not detected / runs on CPU.** Check with
  `python -c "import torch; print(torch.cuda.is_available())"`. If `False` on an
  NVIDIA card, you likely installed a CPU-only torch — reinstall using the
  `--extra-index-url .../cu128` in `requirements.txt`. **RTX 50-series (Blackwell)
  requires the cu128 builds**; older CUDA wheels won't drive it.
- **`torchcodec` DLL / "Could not load libtorchcodec_core*.dll" on Windows.**
  This is a **benign warning** — transcription uses faster-whisper and diarization
  uses an in-memory waveform, so neither needs torchcodec. If it ever becomes a
  hard error, make sure `transformers<5` is installed (5.x couples the audio
  pipeline to torchcodec).
- **`WinError 1314` / "a required privilege is not held"** during model download.
  A HuggingFace cache symlink issue on Windows without Developer Mode. It's
  handled automatically (`HF_HUB_DISABLE_SYMLINKS=1` is set in `src/config.py`);
  if you hit it elsewhere, set that env var before launching.
- **CUDA out of memory (smaller GPUs).** Use a smaller Whisper by adding
  `ASR_MODEL=medium` (or `small`) to `.env`, or force CPU-friendly compute by
  running without a GPU (the app falls back to `int8` on CPU automatically).
- **Transcription is very slow on CPU.** The CPU default is Whisper `small`;
  `ASR_MODEL=base` or `tiny` is faster but less accurate. Leave the translation
  box unticked unless you need it — it doubles the transcription work.
- **`ffmpeg` not found.** No action needed — it's bundled via `imageio-ffmpeg`.
- **401 / gated model error on diarization.** Accept the terms at
  <https://huggingface.co/pyannote/speaker-diarization-3.1> and confirm your
  `HUGGINGFACE_TOKEN` in `.env` is valid.

## Notes

- **GPU is used automatically** when CUDA is available; otherwise it falls back
  to CPU (the app shows which device and Whisper model are active).
- This app is built for **local** use. Hosted Streamlit Community Cloud has no
  GPU and little RAM, so the large Whisper/diarization models won't run there
  without swapping in much smaller models.
- Temporary audio files are written to the OS temp directory (unique names, so
  concurrent runs don't collide) and cleaned up after each analysis.
