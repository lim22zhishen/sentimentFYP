"""Local (transformers) sentiment analysis.

Pure core logic: returns dataclasses and raises on error — no Streamlit.
"""

import re

from src.models import load_sentiment_pipeline
from src.schemas import SentimentResult, Turn

# Normalise model label variants to the app's POSITIVE / NEUTRAL / NEGATIVE.
_LABEL_MAP = {
    "negative": "NEGATIVE",
    "neutral": "NEUTRAL",
    "positive": "POSITIVE",
    "label_0": "NEGATIVE",
    "label_1": "NEUTRAL",
    "label_2": "POSITIVE",
}

# "Speaker: message" or "Speaker:message", split on the first colon. A colon
# followed by a digit or "/" is part of a time ("12:30") or URL ("http://"), not
# a speaker label.
_TURN_RE = re.compile(r"^(?P<speaker>[^:]+?)\s*:(?![\d/])\s*(?P<message>.+)$")
# A speaker label is a short name ("Alice", "Dr. Smith (Cardiology)"), not the
# start of a sentence like "I think the answer is: no".
MAX_SPEAKER_WORDS = 4
MAX_SPEAKER_CHARS = 40


def _normalize_label(label) -> str:
    return _LABEL_MAP.get(str(label).strip().lower(), str(label).strip().upper())


def _to_result(label_scores: list[dict]) -> SentimentResult:
    """Build a result from one input's scores for every label."""
    probs = {_normalize_label(s["label"]): float(s["score"]) for s in label_scores}
    label = max(probs, key=probs.get)
    polarity = round(probs.get("POSITIVE", 0.0) - probs.get("NEGATIVE", 0.0), 2)
    return SentimentResult(
        sentiment=label,
        confidence=round(probs[label], 2),
        polarity=polarity + 0.0,  # turns -0.0 into 0.0 so it doesn't display as "-0.00"
    )


def analyze_sentiment(texts: list[str]) -> list[SentimentResult]:
    """Classify each string in ``texts``.

    Returns a list of :class:`SentimentResult`, one per input.
    """
    items = [t if isinstance(t, str) else "" for t in texts]
    if not items:
        return []

    classifier = load_sentiment_pipeline()
    # top_k=None returns every label's probability (not just the top one), which
    # polarity needs. The result is one list of label scores per input.
    results = classifier(items, truncation=True, batch_size=16, top_k=None)

    # Guard against a single input coming back un-nested.
    if len(items) == 1 and results and isinstance(results[0], dict):
        results = [results]

    return [_to_result(label_scores) for label_scores in results]


def split_conversation(text: str) -> list[Turn]:
    """Split free-form conversation text into per-line :class:`Turn` objects.

    Lines shaped like ``Speaker: message`` (space after the colon optional) are
    split into speaker + message when the label is short enough to be a name;
    other lines are attributed to "Unknown" with the whole line as the message.
    """
    turns = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _TURN_RE.match(line)
        speaker = match["speaker"] if match else ""
        if (
            match
            and len(speaker) <= MAX_SPEAKER_CHARS
            and len(speaker.split()) <= MAX_SPEAKER_WORDS
        ):
            turns.append(Turn(speaker=speaker, message=match["message"]))
        else:
            turns.append(Turn(speaker="Unknown", message=line))
    return turns
