import streamlit as st
import os
import datetime
from transformers import pipeline
import pandas as pd
import plotly.express as px
import whisper
import ffmpeg
import re
import math

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

def align_sentences_with_timestamps(segments, sentences):
    """
    Align Whisper transcription segments with split sentences and extract timestamps.
    """
    aligned_sentences = []
    current_segment_index = 0
    current_start = segments[0]['start'] if segments else 0.0

    for sentence in sentences:
        while current_segment_index < len(segments):
            segment = segments[current_segment_index]
            if sentence in segment['text']:
                # Align the sentence with the segment timestamp
                aligned_sentences.append({
                    "start": round(segment['start'], 2),
                    "end": round(segment['end'], 2),
                    "text": sentence
                })
                current_segment_index += 1
                break
            current_segment_index += 1
        else:
            # If no exact match, estimate the timestamp
            aligned_sentences.append({
                "start": round(current_start, 2),
                "end": round(current_start + 5, 2),  # Assume ~5 seconds duration
                "text": sentence
            })
            current_start += 5

    return aligned_sentences


# Function to split transcription into sentences
def split_into_sentences(transcription):
    # Use regex to split by punctuation, keeping sentence structure
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', transcription)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Highlight positive and negative sentiments
def style_table(row):
    if row["Sentiment"] == "POSITIVE":
        return ['background-color: #d4edda'] * len(row)
    elif row["Sentiment"] == "NEGATIVE":
        return ['background-color: #f8d7da'] * len(row)
    else:
        return [''] * len(row)
                
# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Streamlit app
st.title("Audio Transcription and Sentiment Analysis App")

# Input section for customer service conversation or audio file
input_type = st.radio("Select Input Type", ("Text", "Audio"))

if input_type == "Text":
    st.write("Enter a customer service conversation (each line is a new interaction between customer and service agent):")
    conversation = st.text_area("Conversation", height=300, placeholder="Enter customer-service interaction here...")
elif input_type == "Audio":
    st.write("Upload an audio file (WAV):")
    uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

# Add a button to run the analysis
if st.button('Run Sentiment Analysis'):
    if input_type == "Text" and conversation:
        # Process the text input
        messages = [msg.strip() for msg in conversation.split("\n") if msg.strip()]

        # Limit processing of large conversations (for memory optimization)
        MAX_MESSAGES = 20  # Only process up to 20 messages at once
        if len(messages) > MAX_MESSAGES:
            st.warning(f"Only analyzing the first {MAX_MESSAGES} messages for memory efficiency.")
            messages = messages[:MAX_MESSAGES]

        # Analyze each message for sentiment in batches
        sentiments = batch_analyze_sentiments(messages)

        # Create structured data
        results = []
        for i, msg in enumerate(messages):
            # Split each message into speaker and content
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
        
    if input_type == "Audio":
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            temp_file_path = os.path.join("temp_audio.wav")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())
    
            st.audio(temp_file_path, format='audio/wav')
    
            st.write("Transcribing audio using Whisper...")
            try:
                whisper_model = load_whisper_model()
                result = whisper_model.transcribe(temp_file_path, word_timestamps=True)
                transcription = result["text"]
                segments = result.get("segments", [])
            except Exception as e:
                st.error(f"Whisper transcription failed: {str(e)}")
                transcription = ""
                segments = []
    
            if transcription:
                st.success("Transcription Complete!")
                st.text_area("Transcription", transcription, height=200)
    
                # Split transcription into sentences
                st.write("Splitting sentences with timestamps...")
                sentences = split_into_sentences(transcription)
    
                # Align sentences with timestamps
                aligned_sentences = align_sentences_with_timestamps(segments, sentences)
    
                # Display aligned sentences with timestamps
                aligned_df = pd.DataFrame(aligned_sentences)
                st.write("Aligned Sentences with Timestamps:")
                st.dataframe(aligned_df)
    
                # Analyze sentiment for each sentence
                st.write("Analyzing sentiment...")
                messages = [s["text"] for s in aligned_sentences]
                sentiments = batch_analyze_sentiments(messages)
    
                # Add sentiment analysis to the aligned DataFrame
                for i, sentiment in enumerate(sentiments):
                    aligned_sentences[i]["Sentiment"] = sentiment["label"]
                    aligned_sentences[i]["Score"] = round(sentiment["score"], 2)
    
                # Convert to final DataFrame
                final_df = pd.DataFrame(aligned_sentences)
                styled_df = final_df.style.apply(style_table, axis=1)
    
                # Display results
                st.write("Sentiment Analysis with Timestamps:")
                st.dataframe(styled_df)
    
                # Plot sentiment over time
                fig = px.line(
                    final_df,
                    x='start',
                    y='Score',
                    color='Sentiment',
                    title="Sentiment Score Over Time",
                    markers=True
                )
                fig.update_layout(xaxis_title="Time (Seconds)", yaxis_title="Sentiment Score")
                st.plotly_chart(fig)
    
            else:
                st.warning("No transcription available to analyze.")
    
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        else:
            st.info("Please upload a .wav file to start transcription.")
