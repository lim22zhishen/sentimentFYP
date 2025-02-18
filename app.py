import streamlit as st
import os
import datetime
from transformers import pipeline
import pandas as pd
import plotly.express as px
import openai
from openai import OpenAI
import re
from pyannote.audio import Pipeline

HUGGINGFACE_TOKEN = st.secrets['token']
OPENAI_API_KEY = st.secrets["keys"]
openai.api_key = OPENAI_API_KEY

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=OPENAI_API_KEY)


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

# Function to perform transcription using OpenAI Whisper API with updated client
def transcribe_with_openai(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
    
    # Extract word-level timestamps if available
    word_timestamps = []
    
    # Check if the response has segments with word-level timestamps
    if hasattr(response, 'segments') and response.segments:
        for segment in response.segments:
            if hasattr(segment, 'words'):
                for word in segment.words:
                    word_timestamps.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end
                    })
    
    # Get the transcription text and language
    transcription = response.text
    language = getattr(response, 'language', 'en')
    
    return transcription, language, word_timestamps

# Function to split transcription into sentences
def split_into_sentences(transcription):
    # Use regex to split by punctuation, keeping sentence structure
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', transcription)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def align_sentences_with_diarization(sentences, word_timestamps, speaker_segments):
    """
    Align Whisper sentences with PyAnnote speaker diarization results.
    """
    aligned_sentences = []

    # Loop through sentences and estimate speaker
    for sentence in sentences:
        words = sentence.split()
        sentence_start = None
        sentence_end = None

        # Find the start and end timestamps for the current sentence
        # This is a simplification - in practice you might need more sophisticated alignment
        if word_timestamps:
            for word_data in word_timestamps:
                if word_data['word'].strip() in words[0].lower() and sentence_start is None:
                    sentence_start = word_data['start']
                if word_data['word'].strip() in words[-1].lower():
                    sentence_end = word_data['end']
                    break

        # Fallback in case timestamps are missing
        sentence_start = round(sentence_start or 0.0, 2)
        sentence_end = round(sentence_end or sentence_start + 5, 2)

        # Assign speaker based on diarization timestamps
        speaker = "Unknown"
        max_overlap = 0
        
        for segment in speaker_segments:
            # Find the segment with maximum overlap with the sentence
            segment_start, segment_end = segment['start'], segment['end']
            overlap_start = max(sentence_start, segment_start)
            overlap_end = min(sentence_end, segment_end)
            
            if overlap_end > overlap_start:  # If there's an overlap
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    speaker = segment['speaker']

        aligned_sentences.append({
            "start": sentence_start,
            "end": sentence_end,
            "text": sentence,
            "speaker": speaker
        })

    return aligned_sentences

# Highlight positive and negative sentiments
def style_table(row):
    if row["Sentiment"] == "POSITIVE":
        return ['background-color: #d4edda'] * len(row)
    elif row["Sentiment"] == "NEGATIVE":
        return ['background-color: #f8d7da'] * len(row)
    else:
        return [''] * len(row)
                
# Load PyAnnote diarization pipeline
@st.cache_resource
def load_diarization_pipeline():
    # Use your Hugging Face token here
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_TOKEN)
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
        
    elif input_type == "Audio" and uploaded_file:
        temp_file_path = "temp_audio.wav"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.audio(temp_file_path, format="audio/wav")
        
        # Transcribe with OpenAI Whisper API
        st.write("Transcribing with OpenAI Whisper...")
        transcription, detected_language, word_timestamps = transcribe_with_openai(temp_file_path)
        st.write(f"Detected Language: **{detected_language}**")
        st.write("Transcription:")
        st.text_area("Transcript", transcription, height=200)

        # Speaker Diarization
        st.write("Performing Speaker Diarization...")
        diarization_pipeline = load_diarization_pipeline()
        speaker_segments = diarize_audio(diarization_pipeline, temp_file_path)

        # Align sentences with speakers
        st.write("Aligning transcription with speaker labels...")
        sentences = split_into_sentences(transcription)
        sentences_with_speakers = align_sentences_with_diarization(sentences, word_timestamps, speaker_segments)

        # Sentiment Analysis
        st.write("Performing Sentiment Analysis...")
        messages = [s["text"] for s in sentences_with_speakers]
        sentiments = batch_analyze_sentiments(messages)

        # Create results DataFrame
        results = []
        for i, sentiment in enumerate(sentiments):
            speaker = sentences_with_speakers[i]["speaker"]
            results.append({
                "Speaker": speaker,
                "Text": messages[i],
                "Sentiment": sentiment["label"],
                "Score": round(sentiment["score"], 2)
            })

        df = pd.DataFrame(results)
        st.write("Final Analysis:")
        st.dataframe(df)

        # Sentiment visualization
        fig = px.line(df, x=df.index, y="Score", color="Sentiment", title="Sentiment Score Over Time")
        st.plotly_chart(fig)

        # Clean up
        os.remove(temp_file_path)
    else:
        st.warning("Please provide valid input.")
