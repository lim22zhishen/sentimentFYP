"""Presentation/formatting helpers for the Streamlit UI.

These wrap the repeated "show a sentiment results table / chart / download"
steps so the Text and Audio flows in ``app.py`` share one render path.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

# Map the POSITIVE / NEUTRAL / NEGATIVE labels to numeric values for plotting.
SENTIMENT_MAP = {"positive": 1, "neutral": 0, "negative": -1}


def style_table(row):
    """Highlight rows green for POSITIVE and red for NEGATIVE sentiment."""
    if row["Sentiment"] == "POSITIVE":
        return ['background-color: #d4edda'] * len(row)
    elif row["Sentiment"] == "NEGATIVE":
        return ['background-color: #f8d7da'] * len(row)
    else:
        return [''] * len(row)


def style_y_axis(fig):
    """Show sentiment values as labels on the y-axis."""
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=[-1, 0, 1],
            ticktext=['Negative', 'Neutral', 'Positive'],
        )
    )


def render_sentiment_table(df):
    """Format numeric columns and render the styled sentiment table."""
    display = df.copy()
    display["Score"] = display["Score"].apply(lambda x: f"{float(x):.2f}")
    for col in ("Start Time", "End Time"):
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{float(x):.2f}")
    st.dataframe(display.style.apply(style_table, axis=1))


def render_sentiment_chart(df, x_col, x_title):
    """Plot sentiment (mapped to -1/0/1) over ``x_col``, colored by speaker."""
    plot_df = df.copy()
    plot_df["SentimentValue"] = plot_df["Sentiment"].str.lower().map(SENTIMENT_MAP)
    fig = px.line(
        plot_df, x=x_col, y="SentimentValue", color="Speaker",
        title="Sentiment Changes Over Time", markers=True,
        labels={x_col: x_title, "SentimentValue": "Sentiment"},
    )
    style_y_axis(fig)
    fig.update_traces(marker=dict(size=10))
    st.plotly_chart(fig)


def download_results_button(df, filename="sentiment_results.csv"):
    """Offer the results table as a CSV download."""
    st.download_button(
        "Download results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )
