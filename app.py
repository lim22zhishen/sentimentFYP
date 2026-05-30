import os
import datetime

import pandas as pd
import streamlit as st

from src.config import DEVICE
from src.sentiment import analyze_sentiment, split_conversation
from src.audio import (
    transcribe_audio,
    diarize_audio,
    assign_speakers_to_sentences,
    process_audio_file,
)
from src.models import load_diarization_pipeline
from src.ui_helpers import (
    render_sentiment_table,
    render_sentiment_chart,
    render_speaker_summary,
    render_export_buttons,
)

AUDIO_MIME = {
    ".mp3": "audio/mp3",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".m4a": "audio/mp4",
}


def run_text_analysis(conversation):
    """Analyze a pasted conversation. Returns a render-ready results dict."""
    turns = split_conversation(conversation)
    if not turns:
        st.warning("No text to analyze.")
        return None

    sentiments = analyze_sentiment([t["message"] for t in turns])
    base_time = datetime.datetime.now()

    rows = []
    for i, (turn, sentiment) in enumerate(zip(turns, sentiments)):
        rows.append({
            "Timestamp": (base_time + datetime.timedelta(seconds=i * 10)).strftime('%Y-%m-%d %H:%M:%S'),
            "Speaker": turn["speaker"],
            "Message": turn["message"],
            "Sentiment": sentiment["sentiment"],
            "Score": round(sentiment["confidence"], 2),
        })

    return {"mode": "text", "df": pd.DataFrame(rows)}


def run_audio_analysis(uploaded_file):
    """Transcribe, diarize and analyze an uploaded audio file.

    Heavy work runs inside a single ``st.status`` block; failures surface as an
    error (and return ``None``) rather than producing fabricated results. The
    result dict is returned so the caller can persist and re-render it without
    recomputing.
    """
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    audio_bytes = uploaded_file.getvalue()
    temp_file_path = None

    try:
        with st.status("Analyzing audio…", expanded=True) as status:
            st.write("Preparing audio…")
            temp_file_path = process_audio_file(uploaded_file)

            st.write("Transcribing speech…")
            audio_results = transcribe_audio(temp_file_path)

            st.write("Identifying speakers…")
            diarization_pipeline = load_diarization_pipeline()
            speaker_segments = diarize_audio(diarization_pipeline, temp_file_path)

            st.write("Aligning transcription with speakers…")
            sentences_with_speakers = assign_speakers_to_sentences(
                audio_results, speaker_segments
            )

            if not sentences_with_speakers:
                status.update(label="No transcribed segments to analyze.", state="error")
                return None

            st.write("Scoring sentiment…")
            messages = [s["text"] for s in sentences_with_speakers]
            sentiments = analyze_sentiment(messages)
            status.update(label="Analysis complete", state="complete", expanded=False)
    except Exception as e:
        st.error(f"Audio processing failed: {e}")
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    rows = []
    for i, sentiment in enumerate(sentiments):
        rows.append({
            "Speaker": sentences_with_speakers[i]["speaker"],
            "Text": messages[i],
            "Sentiment": sentiment["sentiment"],
            "Score": round(sentiment["confidence"], 2),
            "Start Time": sentences_with_speakers[i]["start"],
            "End Time": sentences_with_speakers[i]["end"],
        })

    df = pd.DataFrame(rows)
    df["Mid Time"] = df[["Start Time", "End Time"]].astype(float).mean(axis=1)

    return {
        "mode": "audio",
        "df": df,
        "language": audio_results["primary_language"],
        "transcription": audio_results["transcription"],
        "translation": audio_results["translation"],
        "audio_bytes": audio_bytes,
        "audio_format": AUDIO_MIME.get(file_extension, "audio/wav"),
    }


def render_results(data):
    """Render a results dict produced by run_text_analysis / run_audio_analysis."""
    df = data["df"]

    if data["mode"] == "audio":
        st.audio(data["audio_bytes"], format=data["audio_format"])
        st.write(f"Primary Language: **{data['language']}**")

        st.write("### Original Transcription:")
        st.text_area("Transcript", data["transcription"], height=200)

        if data["translation"]:
            st.write("### English Translation:")
            st.text_area("Translation", data["translation"], height=200)

        st.write("### Final Analysis")
        render_sentiment_table(df)
        render_speaker_summary(df)
        render_sentiment_chart(df, x_col="Mid Time", x_title="Time (s)")
    else:
        st.write("### Conversation with Sentiment Labels")
        render_sentiment_table(df)
        render_speaker_summary(df)
        render_sentiment_chart(df, x_col="Timestamp", x_title="Timestamp")

    render_export_buttons(data)


# Streamlit app
st.title("Sentiment Analysis App")
st.caption(f"Running fully locally on **{DEVICE.upper()}** — no external API.")

input_type = st.radio("Select Input Type", ("Text", "Audio"), index=0)

conversation = None
uploaded_file = None

if input_type == "Text":
    st.write("Enter text:")
    conversation = st.text_area("Conversation", height=300, placeholder="Enter text here...")
elif input_type == "Audio":
    st.write("Upload an audio file :")
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg", "m4a", "flac"])

# Run analysis on click and stash the result; rendering reads from session state
# so widget interactions don't re-run the (expensive) pipeline.
if st.button("Run Sentiment Analysis"):
    if input_type == "Text" and conversation:
        st.session_state["results"] = run_text_analysis(conversation)
    elif input_type == "Audio" and uploaded_file:
        st.session_state["results"] = run_audio_analysis(uploaded_file)
    else:
        st.warning("Provide some input first.")

if st.session_state.get("results"):
    render_results(st.session_state["results"])
elif "results" in st.session_state:
    st.info("Analysis finished but produced no results — check the messages above.")
