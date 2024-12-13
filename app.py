import streamlit as st
import os
import datetime
from transformers import pipeline
import pandas as pd
import plotly.express as px
import whisper
import ffmpeg
import re

# Use a smaller and lighter model (distilbert instead of XLM-Roberta)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to scale sentiment scores
def scale_score(label, score):
    return 5 * (score - 0.5) / 0.5 if label == "POSITIVE" else -5 * (1 - score) / 0.5

# Function to analyze sentiment in batches
def batch_analyze_sentiments(messages):
    results = sentiment_pipeline(messages)
    sentiments = [
        {"label": res["label"], "score": scale_score(res["label"], res["score"])}
        for res in results
    ]
    return sentiments

# Function to split transcription into sentences
def split_into_sentences(transcription):
    # Use regex to split by punctuation, keeping sentence structure
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', transcription)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Streamlit app
st.title("Audio Transcription and Sentiment Analysis App")
st.write("Upload a .wav file, and the app will transcribe the audio and analyze the sentiment of the conversation.")

# File uploader
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join("temp_audio.wav")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(temp_file_path, format='audio/wav')

    st.write("Transcribing audio using Whisper...")
    try:
        whisper_model = load_whisper_model()
        result = whisper_model.transcribe(temp_file_path)
        transcription = result["text"]
    except Exception as e:
        st.error(f"Whisper transcription failed: {str(e)}")
        transcription = ""

    if transcription:
        st.success("Transcription Complete!")
        st.text_area("Transcription", transcription, height=200)

        # Process transcription for sentiment analysis
        st.write("Analyzing sentiment...")
        
        # Split transcription into sentences
        sentences = split_into_sentences(transcription)

        # Limit processing of large transcriptions (for memory optimization)
        MAX_MESSAGES = 20  # Only process up to 20 messages at once
        if len(messages) > MAX_MESSAGES:
            st.warning(f"Only analyzing the first {MAX_MESSAGES} messages for memory efficiency.")
            messages = messages[:MAX_MESSAGES]

        # Analyze each message for sentiment in batches
        sentiments = batch_analyze_sentiments(sentences)

        # Create structured data
        results = []
        for i, msg in enumerate(messages):
            # Split each message into speaker and content if possible
            if ": " in msg:
                speaker, content = msg.split(": ", 1)
            else:
                speaker, content = "Unknown", msg

            sentiment = sentiments[i]
            results.append({
                "Timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Speaker": speaker,
                "Message": content,
                "Sentiment": sentiment["label"],
                "Score": round(sentiment["score"], 2)
            })

        # Convert the results into a DataFrame
        df = pd.DataFrame(results)

        # Highlight positive and negative sentiments
        def style_table(row):
            if row["Sentiment"] == "POSITIVE":
                return ['background-color: #d4edda'] * len(row)
            elif row["Sentiment"] == "NEGATIVE":
                return ['background-color: #f8d7da'] * len(row)
            else:
                return [''] * len(row)

        styled_df = df.style.apply(style_table, axis=1)

        # Display the DataFrame
        st.write("Conversation with Sentiment Labels:")
        st.dataframe(styled_df)

        # Plot sentiment over time using Plotly
        fig = px.line(
            df,
            x='Timestamp',
            y='Score',
            color='Sentiment',
            title="Sentiment Score Over Time",
            markers=True
        )
        fig.update_traces(marker=dict(size=10))
        st.plotly_chart(fig)

    else:
        st.warning("No transcription available to analyze.")

    # Clean up the temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
else:
    st.info("Please upload a .wav file to start transcription.")
