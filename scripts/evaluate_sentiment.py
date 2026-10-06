"""Measure the sentiment model against a labelled CSV.

The CSV needs a text column and a label column (negative / neutral / positive,
or neg / neu / pos). Run from the project root, inside the app's virtualenv:

    python scripts/evaluate_sentiment.py scripts/sample_sentiment.csv
    python scripts/evaluate_sentiment.py data.csv --text-col sentence --label-col sentiment
    python scripts/evaluate_sentiment.py data.csv --errors misclassified.csv

Prints accuracy, macro F1, per-label precision / recall / F1 and a confusion
matrix. ``--errors`` saves the misclassified rows for error analysis.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Let "python scripts/evaluate_sentiment.py" find the src package in the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import evaluate, format_report, normalize_gold_label  # noqa: E402
from src.sentiment import analyze_sentiment  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the sentiment model on a labelled CSV."
    )
    parser.add_argument("csv", help="CSV file with a text column and a label column")
    parser.add_argument("--text-col", default="text", help="text column (default: text)")
    parser.add_argument("--label-col", default="label", help="label column (default: label)")
    parser.add_argument("--errors", metavar="PATH", help="write misclassified rows to this CSV")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)
    missing = [c for c in (args.text_col, args.label_col) if c not in df.columns]
    if missing:
        parser.error(f"column(s) {missing} not in {args.csv}; found {list(df.columns)}")
    df = df.dropna(subset=[args.text_col, args.label_col]).reset_index(drop=True)

    try:
        gold = [normalize_gold_label(v) for v in df[args.label_col]]
    except ValueError as e:
        parser.error(str(e))

    results = analyze_sentiment(df[args.text_col].astype(str).tolist())
    predicted = [r.sentiment for r in results]
    print(format_report(evaluate(gold, predicted)))

    if args.errors:
        scored = df.assign(
            gold=gold,
            predicted=predicted,
            confidence=[r.confidence for r in results],
            polarity=[r.polarity for r in results],
        )
        errors = scored[scored["gold"] != scored["predicted"]]
        errors.to_csv(args.errors, index=False)
        print(f"\n{len(errors)} misclassified row(s) written to {args.errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
