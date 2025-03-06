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

def remove_first_two_sentences(text):
    # Split sentences based on punctuation (better than just splitting by newlines)
    sentences = re.split(r'(?<=[.!?])\s+', text, maxsplit=1)
    
    if len(sentences) > 2:
        return sentences[2]  # Return everything except the first sentence
    return text  # Return original if only one sentence exists


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
    Perform speaker diarization with improved speaker label handling.
    """
    try:
        st.write("Starting diarization...")
        diarization_result = diarization_pipeline(audio_file)
        
        speaker_segments = []
        unique_speakers = set()
        
        for segment, _, speaker in diarization_result.itertracks(yield_label=True):
            # Store the exact speaker label from PyAnnote
            speaker_id = speaker  # Keep original ID (like "SPEAKER_00", "SPEAKER_01")
            unique_speakers.add(speaker_id)
            
            segment_info = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "speaker": speaker_id
            }
            speaker_segments.append(segment_info)
    
        st.success(f"Diarization complete. Found {len(unique_speakers)} unique speakers and {len(speaker_segments)} speech segments.")
        
        # Map speaker IDs to more user-friendly names (optional)
        speaker_map = {}
        for i, speaker in enumerate(sorted(unique_speakers)):
            speaker_map[speaker] = f"Speaker {i+1}"
        
        
        # Map the speaker IDs in the segments (optional)
        mapped_segments = []
        for segment in speaker_segments:
            mapped_segment = segment.copy()
            mapped_segment["original_speaker"] = segment["speaker"]
            mapped_segment["speaker"] = speaker_map.get(segment["speaker"], segment["speaker"])
            mapped_segments.append(mapped_segment)

        return mapped_segments
        
    except Exception as e:
        st.error(f"Speaker diarization failed: {str(e)}")
        return [{"start": 0.0, "end": 1000.0, "speaker": "Speaker 1"}]

def handle_multilanguage_audio(audio_file_path, target_language="english"):
    """
    Handle audio that may contain multiple languages:
    1. Transcribe using Whisper (which can handle multiple languages)
    2. Detect language segments if needed
    3. Translate each segment appropriately
    4. Return sentence-level timestamps instead of word timestamps
    """
    # Initial transcription using Whisper API
    with open(audio_file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
    
    transcription = response.text
    primary_language = getattr(response, 'language', 'unknown')

    # Extract sentence-level (segment) timestamps
    sentence_timestamps = []
    if hasattr(response, 'segments') and response.segments:
        for segment in response.segments:
            sentence_timestamps.append({
                "text": segment.text,
                "start": segment.start,
                "end": segment.end
            })
    
    # Analyze for potential multiple languages
    try:
        language_analysis = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Analyze if this text contains multiple languages. If it does, identify which languages and approximately which parts are in which language."},
                {"role": "user", "content": transcription}
            ]
        )
        language_analysis_result = language_analysis.choices[0].message.content
        
        multiple_languages_detected = "multiple languages" in language_analysis_result.lower() or "different languages" in language_analysis_result.lower()
        
        if multiple_languages_detected:
            st.info(language_analysis_result)
    except Exception as e:
        st.warning(f"Language analysis failed: {e}")
        multiple_languages_detected = False
    
    # Translate if needed
    translated_text = None
    if primary_language != 'en':
        try:
            translation_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Translate the following text to English. If there are multiple languages present, identify each language and translate all of it."},
                    {"role": "user", "content": transcription}
                ]
            )
            translated_text = translation_response.choices[0].message.content
        except Exception as e:
            st.warning(f"Translation failed: {e}")
    
    return {
        "transcription": transcription,
        "primary_language": primary_language,
        "multiple_languages_detected": multiple_languages_detected,
        "language_analysis": language_analysis_result if multiple_languages_detected else None,
        "translation": translated_text,
        "sentence_timestamps": sentence_timestamps  # Returns segment-level timestamps
    }


def split_into_sentences(transcription, word_timestamps=None):
    """
    Split transcription into sentences without relying on punctuation.
    Uses pauses in speech and natural language boundaries.
    
    Args:
        transcription: The text transcription
        word_timestamps: Optional list of word timing data from Whisper
    
    Returns:
        List of sentence strings
    """
    # If we have word timestamps, use pauses to determine sentence boundaries
    if word_timestamps and len(word_timestamps) > 1:
        sentences = []
        current_sentence = []
        
        # Define pause threshold (in seconds) that likely indicates sentence boundary
        PAUSE_THRESHOLD = 0.5
        
        for i in range(len(word_timestamps) - 1):
            current_word = word_timestamps[i]
            next_word = word_timestamps[i+1]
            
            # Add current word to the sentence
            current_sentence.append(current_word["word"].strip())
            
            # Check if there's a significant pause between this word and the next
            pause_duration = next_word["start"] - current_word["end"]
            
            # If significant pause or ending punctuation, end the sentence
            if (pause_duration > PAUSE_THRESHOLD or 
                current_word["word"].strip().endswith((".", "!", "?", "。", "！", "？"))):
                sentences.append(" ".join(current_sentence))
                current_sentence = []
        
        # Add the last word and any remaining words to the last sentence
        if current_sentence or word_timestamps:
            current_sentence.append(word_timestamps[-1]["word"].strip())
            sentences.append(" ".join(current_sentence))
            
        return sentences
    
    # Fallback method without timestamps - use a simple length-based approach
    else:
        # For languages without clear word boundaries (like Chinese/Japanese)
        if any(ord(c) > 0x4e00 and ord(c) < 0x9FFF for c in transcription):
            # Split every ~15 characters for logographic scripts
            char_chunks = [transcription[i:i+15] for i in range(0, len(transcription), 15)]
            return char_chunks
        else:
            # Split by words for languages with spaces
            words = transcription.split()
            # Group into chunks of approximately 10-15 words
            WORDS_PER_SENTENCE = 10
            word_chunks = [words[i:i+WORDS_PER_SENTENCE] for i in range(0, len(words), WORDS_PER_SENTENCE)]
            return [" ".join(chunk) for chunk in word_chunks]


def assign_speakers_to_sentences(audio_results, speaker_segments):
    """Assigns speakers to sentences based on timestamps with improved alignment handling."""
    
    sentence_timestamps = audio_results.get("sentence_timestamps", [])
    translated_texts = audio_results.get("translation", None)
    st.write(translated_texts)
    result = []
    
    if not sentence_timestamps or not speaker_segments:
        return []

    for sentence_info in sentence_timestamps:
        sentence_start = sentence_info["start"]
        sentence_end = sentence_info["end"]
        assigned_speaker = "Unknown Speaker"

        best_overlap = 0  # Track the best overlap duration for speaker assignment

        for speaker_segment in speaker_segments:
            speaker_start = speaker_segment["start"]
            speaker_end = speaker_segment["end"]

            # Check if the sentence falls inside or overlaps with the speaker segment
            if sentence_start <= speaker_end and sentence_end >= speaker_start:
                # Calculate overlap duration
                overlap = min(sentence_end, speaker_end) - max(sentence_start, speaker_start)

                # Assign the speaker with the longest overlap
                if overlap > best_overlap:
                    best_overlap = overlap
                    assigned_speaker = speaker_segment["speaker"]

        # Build the result dictionary
        sentence_entry = {
            "text": sentence_info["text"],
            "start": sentence_start,
            "end": sentence_end,
            "speaker": assigned_speaker,
        }

        # Only include translation if it exists and is a valid list
        if translated_texts and isinstance(translated_texts, list) and i < len(translated_texts):
            sentence_entry["translation"] = translated_texts[i]

        result.append(sentence_entry)
    
    return result


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
input_type = st.radio("Select Input Type", ("Text", "Audio"), index=1)  # 0 = "Text", 1 = "Audio"


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
        
        # Process audio with enhanced multilanguage capability
        st.write("Processing audio...")
        audio_results = handle_multilanguage_audio(temp_file_path)

        # Display language information
        st.write(f"Primary Language: **{audio_results['primary_language']}**")
        
        if audio_results['multiple_languages_detected']:
            st.write("### Multiple Languages Detected")
        
        # Display original transcription
        st.write("### Original Transcription:")
        st.text_area("Transcript", audio_results['transcription'], height=200)
        
        # Display translation if available
        if audio_results['translation']:
            st.write("### English Translation:")
            st.text_area("Translation", remove_first_two_sentences(audio_results['translation']), height=200)

        # Speaker Diarization with improved function
        st.write("Performing Speaker Diarization...")

        try:
            diarization_pipeline = load_diarization_pipeline()
            speaker_segments = diarize_audio(diarization_pipeline, temp_file_path)

            # Align sentences with speakers
            st.write("Aligning transcription with speaker labels...")
            sentences_with_speakers = assign_speakers_to_sentences(audio_results, speaker_segments)

            st.json(sentences_with_speakers)
            # Sentiment Analysis
            st.write("Performing Sentiment Analysis...")
            messages = [s["text"] for s in sentences_with_speakers]

            text_for_analysis = [s["translation"] if "translation" in s else s["text"] for s in sentences_with_speakers]
            sentiments = batch_analyze_sentiments(text_for_analysis)
            
            # When preparing the final results DataFrame
            results = []
            for i, sentiment in enumerate(sentiments):
                if i < len(sentences_with_speakers):  # Safety check
                    speaker = sentences_with_speakers[i]["speaker"]
                    
                    # Add debugging info right in the results
                    results.append({
                        "Speaker": speaker,
                        "Text": messages[i],
                        "Sentiment": sentiment["label"],
                        "Score": round(sentiment["score"], 2),
                        "Start Time": sentences_with_speakers[i]["start"],
                        "End Time": sentences_with_speakers[i]["end"]
                    })
            
            # Let's also check the unique speakers in our results
            unique_result_speakers = set(r["Speaker"] for r in results)
            st.write(f"Unique speakers in final results: {', '.join(unique_result_speakers)}")
            
            df = pd.DataFrame(results)
            styled_df = df.style.apply(style_table, axis=1)
            st.write("Final Analysis:")
            st.dataframe(styled_df)
            
            # For your visualization:
            fig = px.line(df, x=df.index, y="Score", color="Speaker", title="Sentiment Score Over Time")
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"An error occurred during audio processing: {str(e)}")
            st.write("Attempting sentiment analysis on whole transcript without speaker diarization...")
            
            # Fallback: analyze whole transcript
            sentences = split_into_sentences(text_for_analysis)
            sentiments = batch_analyze_sentiments(sentences)
            
            results = []
            for i, sentence in enumerate(sentences):
                sentiment = sentiments[i]
                results.append({
                    "Text": sentence,
                    "Sentiment": sentiment["label"],
                    "Score": round(sentiment["score"], 2)
                })

            df = pd.DataFrame(results)
            styled_df = df.style.apply(style_table, axis=1)
            st.write("Basic Sentiment Analysis (without speaker identification):")
            st.dataframe(styled_df)
    
        # Clean up
        os.remove(temp_file_path)
