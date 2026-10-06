"""Presentation/formatting helpers for the Streamlit UI.

These wrap the repeated "show a sentiment results table / chart / export" steps
so the Text and Audio flows in ``app.py`` share one render path.
"""

import math

import pandas as pd
import plotly.express as px
import streamlit as st


def style_table(row):
    """Highlight rows green for POSITIVE and red for NEGATIVE sentiment."""
    if row["Sentiment"] == "POSITIVE":
        return ['background-color: #d4edda'] * len(row)
    elif row["Sentiment"] == "NEGATIVE":
        return ['background-color: #f8d7da'] * len(row)
    else:
        return [''] * len(row)


def style_y_axis(fig):
    """Label the polarity axis: -1 Negative, 0 Neutral, +1 Positive."""
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=[-1, 0, 1],
            ticktext=['Negative', 'Neutral', 'Positive'],
            range=[-1.2, 1.2],
        ),
        hovermode="x unified",
    )


def render_sentiment_table(df: pd.DataFrame) -> None:
    """Format numeric columns and render the styled sentiment table."""
    if df.empty:
        st.info("No rows to display.")
        return
    display = df.copy()
    display["Score"] = display["Score"].apply(lambda x: f"{float(x):.2f}")
    display["Polarity"] = display["Polarity"].apply(lambda x: f"{float(x):+.2f}")
    for col in ("Start Time", "End Time", "Mid Time"):
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{float(x):.2f}")
    st.dataframe(display.style.apply(style_table, axis=1))


def render_sentiment_chart(df: pd.DataFrame, x_col: str, x_title: str) -> None:
    """Plot polarity over ``x_col``, colored by speaker."""
    if df.empty:
        return
    fig = px.line(
        df, x=x_col, y="Polarity", color="Speaker",
        title="Sentiment Changes Over Time", markers=True,
        hover_data=["Sentiment", "Score"],
        labels={x_col: x_title, "Polarity": "Sentiment"},
    )
    style_y_axis(fig)
    if pd.api.types.is_integer_dtype(df[x_col]):
        # Whole-number axis (turns): no "1.5" ticks, at most ~12 labels.
        fig.update_xaxes(tick0=1, dtick=max(1, math.ceil(df[x_col].max() / 12)))
    fig.update_traces(marker=dict(size=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Each point is P(positive) − P(negative): +1 is confidently positive, "
        "−1 confidently negative, and points near 0 are neutral or uncertain."
    )


def speaker_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-speaker line counts, average polarity and mood (most-positive first).

    Pure function (no Streamlit) so it can be unit-tested.
    """
    summary = (
        df.groupby("Speaker")
        .agg(Lines=("Sentiment", "size"), AvgSentiment=("Polarity", "mean"))
        .reset_index()
        .sort_values("AvgSentiment", ascending=False)
    )
    summary["AvgSentiment"] = summary["AvgSentiment"].round(2)
    summary["Mood"] = summary["AvgSentiment"].apply(
        lambda v: "Positive" if v > 0.15 else "Negative" if v < -0.15 else "Neutral"
    )
    return summary


def render_speaker_summary(df: pd.DataFrame) -> None:
    """Show the per-speaker summary table."""
    if df.empty or "Speaker" not in df.columns:
        return
    st.write("Per-speaker summary:")
    st.dataframe(speaker_summary(df), hide_index=True)


def build_transcript(data: dict) -> str:
    """Build a plain-text transcript preserving language, speakers and timestamps.

    Pure function (no Streamlit) so it can be unit-tested. Expects the audio
    result dict shape produced by ``run_audio_analysis``.
    """
    df = data["df"]
    lines = [
        f"Language: {data.get('language', 'unknown')}",
        "",
    ]
    for _, row in df.iterrows():
        start = float(row.get("Start Time", 0.0))
        end = float(row.get("End Time", 0.0))
        lines.append(
            f"[{start:7.2f} - {end:7.2f}] {row['Speaker']} ({row['Sentiment']}): {row['Text']}"
        )
    if data.get("translation"):
        lines += ["", "--- English translation ---", data["translation"]]
    return "\n".join(lines)


def render_export_buttons(data: dict) -> None:
    """Render CSV / JSON (and transcript, for audio) download buttons."""
    df = data["df"]
    if df.empty:
        return
    is_audio = data.get("mode") == "audio"
    cols = st.columns(3 if is_audio else 2)

    cols[0].download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_results.csv",
        mime="text/csv",
    )
    cols[1].download_button(
        "Download JSON",
        data=df.to_json(orient="records", indent=2).encode("utf-8"),
        file_name="sentiment_results.json",
        mime="application/json",
    )
    if is_audio:
        cols[2].download_button(
            "Download transcript",
            data=build_transcript(data).encode("utf-8"),
            file_name="transcript.txt",
            mime="text/plain",
        )
