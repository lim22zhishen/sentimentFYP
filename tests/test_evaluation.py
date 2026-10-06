"""Unit tests for src.evaluation metrics and the evaluate_sentiment script."""

import pytest

import scripts.evaluate_sentiment as script
from src.evaluation import evaluate, format_report, normalize_gold_label
from src.schemas import SentimentResult

POS, NEU, NEG = "POSITIVE", "NEUTRAL", "NEGATIVE"


def test_evaluate_hand_computed_example():
    gold = [POS, POS, NEG, NEU, NEU]
    pred = [POS, NEU, NEG, NEU, POS]
    report = evaluate(gold, pred)

    assert report.n == 5
    assert report.accuracy == pytest.approx(3 / 5)
    # POSITIVE: 1 of 2 predictions right, 1 of 2 found  -> P = R = F1 = 0.5
    # NEGATIVE: perfect                                  -> 1.0
    # NEUTRAL:  1 of 2 predictions right, 1 of 2 found   -> 0.5
    assert report.per_class[POS].precision == pytest.approx(0.5)
    assert report.per_class[POS].recall == pytest.approx(0.5)
    assert report.per_class[NEG].f1 == pytest.approx(1.0)
    assert report.per_class[NEU].support == 2
    assert report.macro_f1 == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert report.confusion[NEU] == {NEG: 0, NEU: 1, POS: 1}


def test_evaluate_macro_f1_ignores_absent_labels():
    report = evaluate([POS, NEG], [POS, NEG])
    assert report.macro_f1 == pytest.approx(1.0)  # NEUTRAL never occurs
    assert report.per_class[NEU].support == 0


@pytest.mark.parametrize("gold,pred", [([], []), ([POS], []), ([POS], ["GOOD"])])
def test_evaluate_rejects_bad_input(gold, pred):
    with pytest.raises(ValueError):
        evaluate(gold, pred)


@pytest.mark.parametrize("raw,expected", [
    ("positive", POS), (" Negative ", NEG), ("NEU", NEU), ("pos", POS), ("LABEL_0", NEG),
])
def test_normalize_gold_label(raw, expected):
    assert normalize_gold_label(raw) == expected


def test_normalize_gold_label_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown label 'good'"):
        normalize_gold_label("good")


def test_format_report_lists_metrics_and_matrix():
    text = format_report(evaluate([POS, NEG], [POS, POS]))
    assert "Accuracy: 0.500" in text
    assert "Confusion matrix" in text
    assert text.splitlines()[-1].split() == [POS, "0", "0", "1"]


def test_script_reports_and_writes_errors(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data.csv"
    data.write_text("sentence,mood\nlovely,pos\nawful,neg\nfine,neu\n", encoding="utf-8")
    errors = tmp_path / "errors.csv"
    # Predicts POSITIVE for everything: 1 of 3 right.
    monkeypatch.setattr(
        script, "analyze_sentiment",
        lambda texts: [SentimentResult(POS, 0.9, 0.8) for _ in texts],
    )
    assert script.main([str(data), "--text-col", "sentence", "--label-col", "mood",
                        "--errors", str(errors)]) == 0

    out = capsys.readouterr().out
    assert "Examples: 3" in out and "Accuracy: 0.333" in out
    assert errors.read_text(encoding="utf-8").count("\n") == 3  # header + 2 wrong rows


def test_script_rejects_missing_column(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("text,label\nhi,pos\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        script.main([str(data), "--label-col", "sentiment"])
