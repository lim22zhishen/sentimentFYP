"""End-to-end tests of the Streamlit app flows (models are stubbed)."""

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from streamlit.testing.v1 import AppTest

import src.models
import src.sentiment as sentiment

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _fake_sentiment_loader():
    def clf(items, **kw):
        return [
            [{"label": "positive", "score": 0.9}, {"label": "negative", "score": 0.1}]
            if "great" in text.lower()
            else [{"label": "negative", "score": 0.8}, {"label": "positive", "score": 0.2}]
            for text in items
        ]
    return clf


def test_text_mode_rows_are_numbered_turns(monkeypatch):
    monkeypatch.setattr(sentiment, "load_sentiment_pipeline", _fake_sentiment_loader)

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.text_area[0].input("Alice: great\nBob: fine\nunlabelled line")
    at.button[0].click()
    at.run()

    assert not at.exception
    df = at.dataframe[0].value
    assert "Timestamp" not in df.columns
    assert list(df["Turn"]) == [1, 2, 3]
    assert list(df["Speaker"]) == ["Alice", "Bob", "Unknown"]
    assert list(df["Polarity"]) == ["+0.80", "-0.60", "-0.60"]


# --- audio flow: Whisper and pyannote stubbed, everything else real ----------

class _Word:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _Segment:
    def __init__(self, text, start, end, words=None):
        self.text, self.start, self.end, self.words = text, start, end, words


class _Info:
    language = "fr"


class _FakeWhisper:
    def transcribe(self, path, **kwargs):
        if kwargs.get("task") == "translate":
            return iter([_Segment(" Is it great? It is bad.", 0.0, 2.0)]), _Info()
        words = [_Word(" C'est", 0.0, 0.4), _Word(" great?", 0.4, 0.9),
                 _Word(" C'est", 1.1, 1.4), _Word(" nul.", 1.4, 2.0)]
        return iter([_Segment(" C'est great? C'est nul.", 0.0, 2.0, words)]), _Info()


class _Track:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _Annotation:
    def itertracks(self, yield_label=True):
        yield _Track(0.0, 1.0), None, "SPEAKER_00"
        yield _Track(1.0, 2.0), None, "SPEAKER_01"


def _wav_bytes(seconds=2.0, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(sr * seconds), dtype="float32"), sr, format="WAV")
    return buf.getvalue()


@pytest.mark.torch
def test_audio_mode_splits_speakers_and_passes_options(monkeypatch):
    diarize_calls = []

    def fake_diarization(payload, **kwargs):
        diarize_calls.append(kwargs)
        return _Annotation()

    monkeypatch.setattr(sentiment, "load_sentiment_pipeline", _fake_sentiment_loader)
    monkeypatch.setattr(src.models, "load_asr_model", lambda: _FakeWhisper())
    monkeypatch.setattr(src.models, "load_diarization_pipeline", lambda: fake_diarization)

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.radio[0].set_value("Audio")
    at.run()
    at.file_uploader[0].upload("talk.flac", _wav_bytes(), "audio/flac")
    at.number_input[0].set_value(2)
    at.checkbox[0].check()
    at.button[0].click()
    at.run()

    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    assert diarize_calls == [{"num_speakers": 2}]

    df = at.dataframe[0].value
    assert list(df["Speaker"]) == ["Speaker 1", "Speaker 2"]
    assert list(df["Text"]) == ["C'est great?", "C'est nul."]
    assert list(df["Sentiment"]) == ["POSITIVE", "NEGATIVE"]

    texts = {t.label: t.value for t in at.text_area}
    assert texts["Translation"] == "Is it great? It is bad."
