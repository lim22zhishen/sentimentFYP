import streamlit as st
import os
import datetime
import pandas as pd
import plotly.express as px
import whisper
import ffmpeg
import re
import math
from pyannote.audio import Pipeline
from langdetect import detect
from transformers import pipeline

HUGGINGFACE_TOKEN = st.secrets['token']

# Use a smaller and lighter model (distilbert instead of XLM-Roberta)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Translation pipeline (Helsinki-NLP models)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Multilingual sentiment analysis with translation fallback
from langdetect import detect
from transformers import pipeline

# Multilingual sentiment analysis pipeline (uses a multilingual model)
multilingual_sentiment_pipeline = pipeline(
    "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Translation pipeline (Helsinki-NLP models)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Multilingual sentiment analysis with translation fallback
def analyze_multilingual_sentiment(texts):
    sentiments = []
    for text in texts:
        language = detect_language(text)
        
        if language == "en":
            # Directly analyze if English
            result = sentiment_pipeline(text)
        else:
            # Translate to English for non-English text
            try:
                translated_text = translator(text, max_length=512)[0]["translation_text"]
                result = sentiment_pipeline(translated_text)
            except Exception as e:
                result = [{"label": "ERROR", "score": 0}]
                st.error(f"Translation failed for text: {text}. Error: {str(e)}")
        
        sentiments.append({
            "original_text": text,
            "language": language,
            "sentiment": result[0]["label"],
            "score": round(result[0]["score"], 2)
        })
    return sentiments

    
# Function to scale sentiment scores
def scale_score(label, score):
    return 5 * (score - 0.5) / 0.5 if label == "POSITIVE" else -5 * (1 - score) / 0.5

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

def align_sentences_with_diarization(sentences, whisper_word_timestamps, speaker_segments):
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
        for word in whisper_word_timestamps:
            if word['word'] == words[0] and sentence_start is None:
                sentence_start = word['start']
            if word['word'] == words[-1]:
                sentence_end = word['end']

        # Fallback in case timestamps are missing
        sentence_start = round(sentence_start or 0.0, 2)
        sentence_end = round(sentence_end or sentence_start + 5, 2)

        # Assign speaker based on diarization timestamps
        speaker = "Unknown"
        for segment in speaker_segments:
            if segment['start'] <= sentence_start <= segment['end']:
                speaker = segment['speaker']
                break

        aligned_sentences.append({
            "start": sentence_start,
            "end": sentence_end,
            "text": sentence,
            "speaker": speaker
        })

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
    
        # Limit processing of large conversations
        MAX_MESSAGES = 30
        if len(messages) > MAX_MESSAGES:
            st.warning(f"Only analyzing the first {MAX_MESSAGES} messages.")
            messages = messages[:MAX_MESSAGES]
    
        # Analyze each message for sentiment
        sentiments = analyze_multilingual_sentiment(messages)
    
        # Create structured data
        results = []
        for i, sentiment in enumerate(sentiments):
            msg = messages[i]
            if ": " in msg:
                speaker, content = msg.split(": ", 1)
            else:
                speaker, content = "Unknown", msg
            
            results.append({
                "Timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Speaker": speaker,
                "Message": sentiment["original_text"],
                "Language": sentiment["language"],
                "Sentiment": sentiment["sentiment"],
                "Score": sentiment["score"]
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
                word_timestamps = [
                    {"word": word["word"].strip(), "start": word["start"], "end": word["end"]}
                    for segment in segments for word in segment["words"]
                ]
            except Exception as e:
                st.error(f"Whisper transcription failed: {str(e)}")
                os.remove(temp_file_path)
                st.stop()
    
            # Diarize audio using PyAnnote
            st.write("Performing speaker diarization...")
            diarization_pipeline = load_diarization_pipeline()
            speaker_segments = diarize_audio(diarization_pipeline, temp_file_path)
    
            # Align sentences with timestamps and speakers
            st.write("Analyzing timestamps...")
            sentences = split_into_sentences(transcription)
            sentences_with_speakers = align_sentences_with_diarization(sentences, word_timestamps, speaker_segments)

            # Analyze sentiment
            st.write("Analyzing sentiment...")
            messages = [s["text"] for s in sentences_with_speakers]
            sentiments = analyze_multilingual_sentiment(messages)
            
            # Combine everything into a final DataFrame
            for i, sentiment in enumerate(sentiments):
                sentences_with_speakers[i]["Sentiment"] = sentiment["sentiment"]
                sentences_with_speakers[i]["Score"] = sentiment["score"]
                sentences_with_speakers[i]["Language"] = sentiment["language"]


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
