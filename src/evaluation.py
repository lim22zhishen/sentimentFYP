"""Accuracy metrics for the sentiment classifier.

Pure Python (no scikit-learn, no Streamlit) so it adds no dependency and can be
unit-tested. Used by ``scripts/evaluate_sentiment.py``.
"""

from dataclasses import dataclass

from src.sentiment import _normalize_label

LABELS = ("NEGATIVE", "NEUTRAL", "POSITIVE")
_GOLD_ALIASES = {"neg": "NEGATIVE", "neu": "NEUTRAL", "pos": "POSITIVE"}


@dataclass
class ClassMetrics:
    """Precision / recall / F1 for one label."""

    precision: float
    recall: float
    f1: float
    support: int  # gold examples with this label


@dataclass
class EvaluationReport:
    """Overall and per-label results of comparing predictions with gold labels."""

    n: int
    accuracy: float
    macro_f1: float  # mean F1 over the labels that occur in gold or predictions
    per_class: dict[str, ClassMetrics]
    confusion: dict[str, dict[str, int]]  # confusion[gold][predicted]


def normalize_gold_label(raw) -> str:
    """Map a dataset's label (``positive``, ``Neg``, ``LABEL_2`` …) to the app's labels.

    Raises ``ValueError`` for anything else, so a typo in the dataset doesn't
    silently count as a wrong prediction.
    """
    key = str(raw).strip().lower()
    label = _GOLD_ALIASES.get(key) or _normalize_label(key)
    if label not in LABELS:
        raise ValueError(
            f"Unknown label {raw!r}: use negative / neutral / positive (or neg / neu / pos)"
        )
    return label


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(gold: list[str], predicted: list[str]) -> EvaluationReport:
    """Compare predicted labels with gold labels (both already normalized)."""
    if len(gold) != len(predicted):
        raise ValueError(f"{len(gold)} gold labels but {len(predicted)} predictions")
    if not gold:
        raise ValueError("Nothing to evaluate")
    unknown = (set(gold) | set(predicted)) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected label(s): {sorted(unknown)}")

    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, predicted):
        confusion[g][p] += 1

    per_class = {}
    for label in LABELS:
        true_positives = confusion[label][label]
        predicted_as_label = sum(confusion[g][label] for g in LABELS)
        support = sum(confusion[label].values())
        precision = _ratio(true_positives, predicted_as_label)
        recall = _ratio(true_positives, support)
        f1 = _ratio(2 * precision * recall, precision + recall)
        per_class[label] = ClassMetrics(precision, recall, f1, support)

    present = [label for label in LABELS if label in set(gold) | set(predicted)]
    return EvaluationReport(
        n=len(gold),
        accuracy=sum(confusion[label][label] for label in LABELS) / len(gold),
        macro_f1=sum(per_class[label].f1 for label in present) / len(present),
        per_class=per_class,
        confusion=confusion,
    )


def format_report(report: EvaluationReport) -> str:
    """Render a report as a plain-text summary, metrics table and confusion matrix."""
    lines = [
        f"Examples: {report.n}",
        f"Accuracy: {report.accuracy:.3f}",
        f"Macro F1: {report.macro_f1:.3f}",
        "",
        f"{'Label':<10}{'Precision':>10}{'Recall':>8}{'F1':>8}{'Support':>9}",
    ]
    for label, m in report.per_class.items():
        lines.append(
            f"{label:<10}{m.precision:>10.3f}{m.recall:>8.3f}{m.f1:>8.3f}{m.support:>9}"
        )
    lines += [
        "",
        "Confusion matrix (rows = gold, columns = predicted)",
        f"{'':<10}" + "".join(f"{label:>10}" for label in LABELS),
    ]
    for gold_label, row in report.confusion.items():
        lines.append(f"{gold_label:<10}" + "".join(f"{row[p]:>10}" for p in LABELS))
    return "\n".join(lines)
