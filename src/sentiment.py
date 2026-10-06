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


def analyze_sentiment(texts: list[str]) -> list[SentimentResult]:
    """Classify each string in ``texts``.

    Returns a list of :class:`SentimentResult`, one per input.
    """
    items = [t if isinstance(t, str) else "" for t in texts]
    if not items:
        return []

    classifier = load_sentiment_pipeline()
    results = classifier(items, truncation=True, batch_size=16)

    # A single input can return a dict rather than a list.
    if isinstance(results, dict):
        results = [results]

    return [
        SentimentResult(
            sentiment=_normalize_label(r["label"]),
            confidence=round(float(r["score"]), 2),
        )
        for r in results
    ]


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
