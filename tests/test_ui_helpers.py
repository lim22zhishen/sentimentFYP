"""Unit tests for src.ui_helpers pure helpers (transcript building, speaker summary)."""

import pandas as pd

from src.ui_helpers import build_transcript, speaker_summary


def test_speaker_summary_averages_polarity_and_sorts():
    df = pd.DataFrame([
        {"Speaker": "Alice", "Sentiment": "POSITIVE", "Polarity": 0.9},
        {"Speaker": "Alice", "Sentiment": "NEUTRAL", "Polarity": 0.1},
        {"Speaker": "Bob", "Sentiment": "NEGATIVE", "Polarity": -0.8},
        {"Speaker": "Cara", "Sentiment": "NEUTRAL", "Polarity": 0.05},
    ])
    summary = speaker_summary(df)
    assert list(summary["Speaker"]) == ["Alice", "Cara", "Bob"]
    assert list(summary["Lines"]) == [2, 1, 1]
    assert list(summary["AvgSentiment"]) == [0.5, 0.05, -0.8]
    assert list(summary["Mood"]) == ["Positive", "Neutral", "Negative"]


def test_build_transcript_preserves_language_speakers_timestamps():
    df = pd.DataFrame([
        {"Speaker": "Speaker 1", "Text": "hello", "Sentiment": "POSITIVE",
         "Score": 0.9, "Start Time": 0.0, "End Time": 1.5},
        {"Speaker": "Speaker 2", "Text": "bonjour", "Sentiment": "NEUTRAL",
         "Score": 0.8, "Start Time": 1.5, "End Time": 3.0},
    ])
    data = {"mode": "audio", "df": df, "language": "fr", "translation": "hi / good day"}
    text = build_transcript(data)

    assert "Language: fr" in text
    assert "Speaker 1" in text and "Speaker 2" in text
    assert "0.00" in text and "1.50" in text
    assert "POSITIVE" in text
    assert "English translation" in text
    assert "hi / good day" in text


def test_build_transcript_omits_translation_when_absent():
    df = pd.DataFrame([
        {"Speaker": "Speaker 1", "Text": "hi", "Sentiment": "POSITIVE",
         "Score": 0.9, "Start Time": 0.0, "End Time": 1.0},
    ])
    data = {"mode": "audio", "df": df, "language": "en", "translation": None}
    text = build_transcript(data)
    assert "English translation" not in text
