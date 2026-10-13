"""Unit tests for src.sentiment (no model downloads — the classifier is stubbed)."""

import pytest

import src.sentiment as sentiment
from src.sentiment import split_conversation, _normalize_label, _to_result, analyze_sentiment
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


@pytest.mark.parametrize("line,expected", [
    ("Alice:hi there", Turn("Alice", "hi there")),
    ("Speaker 1 : hello", Turn("Speaker 1", "hello")),
    ("Dr. Smith (Cardiology): stable", Turn("Dr. Smith (Cardiology)", "stable")),
    # a sentence before the colon is not a speaker name
    ("I think the answer is: no", Turn("Unknown", "I think the answer is: no")),
    # times and URLs are not speaker separators
    ("Meeting moved to 12:30", Turn("Unknown", "Meeting moved to 12:30")),
    ("https://example.com is down", Turn("Unknown", "https://example.com is down")),
    ("Bob: call me at 12:30", Turn("Bob", "call me at 12:30")),
    # a label with nothing after it stays a plain line
    ("Alice:", Turn("Unknown", "Alice:")),
])
def test_split_conversation_speaker_detection(line, expected):
    assert split_conversation(line) == [expected]


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


def _scores(positive, neutral, negative):
    """One input's classifier output with top_k=None: every label's probability."""
    return [
        {"label": "positive", "score": positive},
        {"label": "neutral", "score": neutral},
        {"label": "negative", "score": negative},
    ]


def test_analyze_sentiment_normalizes_and_rounds(monkeypatch):
    def fake_loader():
        return lambda items, **kw: [_scores(0.987, 0.008, 0.005) for _ in items]

    monkeypatch.setattr(sentiment, "load_sentiment_pipeline", fake_loader)
    assert analyze_sentiment(["a", "b"]) == [
        SentimentResult("POSITIVE", 0.99, 0.98),
        SentimentResult("POSITIVE", 0.99, 0.98),
    ]


def test_analyze_sentiment_polarity_uses_all_probabilities(monkeypatch):
    def fake_loader():
        return lambda items, **kw: [
            _scores(0.10, 0.20, 0.70),  # confidently negative
            _scores(0.45, 0.10, 0.45),  # torn between positive and negative
            _scores(0.30, 0.40, 0.30),  # weakly neutral
        ]

    monkeypatch.setattr(sentiment, "load_sentiment_pipeline", fake_loader)
    neg, torn, neu = analyze_sentiment(["x", "y", "z"])
    assert (neg.sentiment, neg.confidence, neg.polarity) == ("NEGATIVE", 0.7, -0.6)
    assert torn.polarity == 0.0
    assert (neu.sentiment, neu.confidence, neu.polarity) == ("NEUTRAL", 0.4, 0.0)


def test_polarity_has_no_negative_zero():
    # -0.0002 rounds to -0.0, which would display as "-0.00"
    result = _to_result(_scores(0.3330, 0.3338, 0.3332))
    assert str(result.polarity) == "0.0"


def test_analyze_sentiment_requests_every_label(monkeypatch):
    captured = {}

    def fake_loader():
        def clf(items, **kw):
            captured.update(kw)
            return [_scores(1.0, 0.0, 0.0) for _ in items]
        return clf

    monkeypatch.setattr(sentiment, "load_sentiment_pipeline", fake_loader)
    analyze_sentiment(["a"])
    assert "top_k" in captured and captured["top_k"] is None


def test_analyze_sentiment_empty_does_not_load_model(monkeypatch):
    def boom():
        raise AssertionError("loader should not be called for empty input")

    monkeypatch.setattr(sentiment, "load_sentiment_pipeline", boom)
    assert analyze_sentiment([]) == []


def test_analyze_sentiment_handles_single_unnested_result(monkeypatch):
    def fake_loader():
        # A flat list of label scores (not wrapped per input), with LABEL_N names.
        return lambda items, **kw: [
            {"label": "LABEL_2", "score": 0.5},
            {"label": "LABEL_1", "score": 0.3},
            {"label": "LABEL_0", "score": 0.2},
        ]

    monkeypatch.setattr(sentiment, "load_sentiment_pipeline", fake_loader)
    assert analyze_sentiment(["only one"]) == [SentimentResult("POSITIVE", 0.5, 0.3)]


def test_analyze_sentiment_coerces_non_strings(monkeypatch):
    captured = {}

    def fake_loader():
        def clf(items, **kw):
            captured["items"] = items
            return [_scores(0.0, 1.0, 0.0) for _ in items]
        return clf

    monkeypatch.setattr(sentiment, "load_sentiment_pipeline", fake_loader)
    analyze_sentiment(["ok", None, 123])
    assert captured["items"] == ["ok", "", ""]
