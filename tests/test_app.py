"""End-to-end test of the Streamlit text flow (sentiment classifier is stubbed)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import src.sentiment as sentiment

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_text_mode_rows_are_numbered_turns(monkeypatch):
    def fake_loader():
        return lambda items, **kw: [{"label": "positive", "score": 0.9} for _ in items]

    monkeypatch.setattr(sentiment, "load_sentiment_pipeline", fake_loader)

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
