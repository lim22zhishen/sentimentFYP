"""Local (transformers) sentiment analysis."""

from src.models import load_sentiment_pipeline

# Normalise model label variants to the app's POSITIVE / NEUTRAL / NEGATIVE.
_LABEL_MAP = {
    "negative": "NEGATIVE",
    "neutral": "NEUTRAL",
    "positive": "POSITIVE",
    "label_0": "NEGATIVE",
    "label_1": "NEUTRAL",
    "label_2": "POSITIVE",
}


def _normalize_label(label):
    return _LABEL_MAP.get(str(label).strip().lower(), str(label).strip().upper())


def analyze_sentiment(texts):
    """Classify each string in ``texts``.

    Returns a list of ``{"sentiment", "confidence"}`` dicts, one per input.
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
        {
            "sentiment": _normalize_label(r["label"]),
            "confidence": round(float(r["score"]), 2),
        }
        for r in results
    ]


def split_conversation(text):
    """Split free-form conversation text into per-line turns.

    Lines shaped like ``Speaker: message`` are split into speaker + message;
    other lines are attributed to "Unknown".
    """
    turns = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ": " in line:
            speaker, message = line.split(": ", 1)
        else:
            speaker, message = "Unknown", line
        turns.append({"speaker": speaker, "message": message})
    return turns
