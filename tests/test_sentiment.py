"""Unit tests for src.sentiment (no model downloads — the classifier is stubbed)."""

import pytest

import src.sentiment as sentiment
from src.sentiment import split_conversation, _normalize_label, analyze_sentiment
from src.schemas import SentimentResult, Turn


def test_split_conversation_basic():
    assert split_conversation("Alice: hi\nBob: hello there\n") == [
        Turn("Alice", "hi"),
        Turn("Bob", "hello there"),
    ]


def test_split_conversation_unlabeled_and_blank_lines():
    assert split_conversation("just a line\n\n   \nNo colon here") == [
        Turn("Unknown", "just a line"),
        Turn("Unknown", "No colon here"),
    ]


def test_split_conversation_splits_on_first_separator_only():
    assert split_conversation("Sue: time is 12: 30 now") == [
        Turn("Sue", "time is 12: 30 now")
    ]


def test_split_conversation_empty():
    assert split_conversation("") == []


@pytest.mark.parametrize("raw,expected", [
    ("positive", "POSITIVE"),
    ("Negative", "NEGATIVE"),
    ("NEUTRAL", "NEUTRAL"),
    ("LABEL_0", "NEGATIVE"),
    ("label_1", "NEUTRAL"),
    ("label_2", "POSITIVE"),
    ("something_else", "SOMETHING_ELSE"),
])
def test_normalize_label(raw, expected):
    assert _normalize_label(raw) == expected


def test_analyze_sentiment_normalizes_and_rounds(monkeypatch):
    def fake_loader():
        return lambda items, **kw: [{"label": "positive", "score": 0.987} for _ in items]

    monkeypatch.setattr(sentiment, "load_sentiment_pipeline", fake_loader)
    assert analyze_sentiment(["a", "b"]) == [
        SentimentResult("POSITIVE", 0.99),
        SentimentResult("POSITIVE", 0.99),
    ]


def test_analyze_sentiment_empty_does_not_load_model(monkeypatch):
    def boom():
        raise AssertionError("loader should not be called for empty input")

    monkeypatch.setattr(sentiment, "load_sentiment_pipeline", boom)
    assert analyze_sentiment([]) == []


def test_analyze_sentiment_handles_single_dict_result(monkeypatch):
    def fake_loader():
        return lambda items, **kw: {"label": "label_2", "score": 0.5}

    monkeypatch.setattr(sentiment, "load_sentiment_pipeline", fake_loader)
    assert analyze_sentiment(["only one"]) == [SentimentResult("POSITIVE", 0.5)]


def test_analyze_sentiment_coerces_non_strings(monkeypatch):
    captured = {}

    def fake_loader():
        def clf(items, **kw):
            captured["items"] = items
            return [{"label": "neutral", "score": 1.0} for _ in items]
        return clf

    monkeypatch.setattr(sentiment, "load_sentiment_pipeline", fake_loader)
    analyze_sentiment(["ok", None, 123])
    assert captured["items"] == ["ok", "", ""]
