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
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.core import Segment
from pyannote.audio import Model

HUGGINGFACE_TOKEN = st.secrets['token']

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

def diarize_audio(diarization_pipeline, audio_file):
    """
    Perform speaker diarization using PyAnnote.
    Returns a list of speaker segments (start time, end time, speaker label).
    """
    diarization_result = diarization_pipeline(audio_file)
    speaker_segments = []

    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_segments.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "speaker": speaker
        })
    return speaker_segments

def align_speakers_to_sentences(speaker_segments, sentences_with_timestamps):
    """
    Align PyAnnote speaker segments to Whisper sentences.
    """
    for sentence in sentences_with_timestamps:
        sentence["speaker"] = "Unknown"
        for seg in speaker_segments:
            if seg["start"] <= sentence["start"] <= seg["end"]:
                sentence["speaker"] = seg["speaker"]
                break
    return sentences_with_timestamps
    
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

# Load PyAnnote diarization pipeline
@st.cache_resource
def load_diarization_pipeline():
    # Use your Hugging Face token here
    pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_TOKEN)
    return pipeline
    
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
        
    elif input_type == "Audio":
        if uploaded_file is not None:
            # Save the uploaded file
            temp_file_path = "temp_audio.wav"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())
    
            st.audio(temp_file_path, format='audio/wav')
    
            # Transcribe the audio using Whisper
            st.write("Transcribing audio using Whisper...")
            try:
                whisper_model = load_whisper_model()
                result = whisper_model.transcribe(temp_file_path, word_timestamps=True)
                transcription = result["text"]
                segments = result.get("segments", [])
            except Exception as e:
                st.error(f"Whisper transcription failed: {str(e)}")
                os.remove(temp_file_path)
                st.stop()
    
            # Diarize audio using PyAnnote
            st.write("Performing speaker diarization...")
            diarization_pipeline = load_diarization_pipeline()
            speaker_segments = diarize_audio(diarization_pipeline, temp_file_path)
    
            # Align sentences with timestamps and speakers
            sentences = split_into_sentences(transcription)
            sentences_with_timestamps = align_sentences_with_timestamps(segments, sentences)
            sentences_with_speakers = align_speakers_to_sentences(speaker_segments, sentences_with_timestamps)
    
            # Analyze sentiment
            st.write("Analyzing sentiment...")
            messages = [s["text"] for s in sentences_with_speakers]
            sentiments = batch_analyze_sentiments(messages)
    
            # Combine everything into a final DataFrame
            for i, sentiment in enumerate(sentiments):
                sentences_with_speakers[i]["Sentiment"] = sentiment["label"]
                sentences_with_speakers[i]["Score"] = round(sentiment["score"], 2)
    
            final_df = pd.DataFrame(sentences_with_speakers)
    
            # Display results
            st.write("Speaker-Diarized Sentiment Analysis:")
            styled_df = final_df.style.apply(style_table, axis=1)
            st.dataframe(styled_df)
    
            # Visualization
            fig = px.line(
                final_df,
                x='start',
                y='Score',
                color='speaker',
                title="Sentiment Score Over Time by Speaker",
                markers=True
            )
            fig.update_layout(xaxis_title="Time (Seconds)", yaxis_title="Sentiment Score")
            st.plotly_chart(fig)
    
            # Clean up
            os.remove(temp_file_path)
        else:
            st.info("Please upload a .wav file to start transcription.")
