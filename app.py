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
    Perform speaker diarization with improved speaker label handling.
    """
    try:
        st.write("Starting diarization...")
        diarization_result = diarization_pipeline(audio_file)
        
        speaker_segments = []
        unique_speakers = set()
        
        # Debug: Print raw diarization results
        st.write("Raw diarization segments:")
        debug_segments = []
        
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
            debug_segments.append(f"{segment.start:.2f}-{segment.end:.2f}: {speaker_id}")
        
        # Display debug info
        st.code("\n".join(debug_segments[:10]) + 
                (f"\n... plus {len(debug_segments)-10} more segments" if len(debug_segments) > 10 else ""))
        
        st.success(f"Diarization complete. Found {len(unique_speakers)} unique speakers and {len(speaker_segments)} speech segments.")
        
        # Map speaker IDs to more user-friendly names (optional)
        speaker_map = {}
        for i, speaker in enumerate(sorted(unique_speakers)):
            speaker_map[speaker] = f"Speaker {i+1}"
        
        # Show speaker mapping
        st.write("Speaker mapping:")
        for original, mapped in speaker_map.items():
            st.write(f"{original} → {mapped}")
        
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
        import traceback
        st.code(traceback.format_exc())
        return [{"start": 0.0, "end": 1000.0, "speaker": "Speaker 1"}]

def handle_multilanguage_audio(audio_file_path, target_language="english"):
    """
    Handle audio that may contain multiple languages:
    1. Transcribe using Whisper (which can handle multiple languages)
    2. Detect language segments if needed
    3. Translate each segment appropriately
    """
    # Initial transcription
    with open(audio_file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
    
    transcription = response.text
    primary_language = getattr(response, 'language', 'unknown')
    
    # Extract word-level timestamps
    word_timestamps = []
    if hasattr(response, 'segments') and response.segments:
        for segment in response.segments:
            if hasattr(segment, 'words'):
                for word in segment.words:
                    word_timestamps.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end
                    })
    
    # Analyze for potential multiple languages (can be enhanced with more sophisticated detection)
    try:
        # Use GPT to detect if there might be multiple languages
        language_analysis = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Analyze if this text contains multiple languages. If it does, identify which languages and approximately which parts are in which language."},
                {"role": "user", "content": transcription}
            ]
        )
        language_analysis_result = language_analysis.choices[0].message.content
        
        # Check if multiple languages were detected
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
            translation_prompt = "Translate the following text to English. If there are multiple languages present, identify each language and translate all of it."
            translation_response = client.chat.completions.create(
                model="gpt-4",  # Using a more powerful model for multilingual translation
                messages=[
                    {"role": "system", "content": translation_prompt},
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
        "word_timestamps": word_timestamps
    }

def assign_speakers_to_sentences(transcription_segments, speaker_segments):
    """
    Match transcription sentences with speaker segments based on overlapping timestamps.

    Args:
        transcription_segments (list): List of transcription sentences with start and end times.
        speaker_segments (list): List of speaker segments with start and end times.

    Returns:
        list: List of sentences with assigned speakers.
    """
    results = []
    for segment in transcription_segments:
        sentence_start = segment["start"]
        sentence_end = segment["end"]
        text = segment["text"]

        # Default speaker is "Unknown"
        speaker = "Unknown"

        # Check for overlap with speaker segments
        for speaker_segment in speaker_segments:
            if sentence_start < speaker_segment["End"] and sentence_end > speaker_segment["Start"]:
                speaker = speaker_segment["Speaker"]
                break  # Assign the first matching speaker

        results.append({
            "Speaker": speaker,
            "Text": text,
            "Start": sentence_start,
            "End": sentence_end
        })
    return results
    
def split_into_sentences(transcription, word_timestamps):
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
        PAUSE_THRESHOLD = 0.7
        
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
            
def align_sentences_with_diarization(sentences, word_timestamps, speaker_segments):
    """
    Improved alignment that focuses on precise time matching between diarization and transcription.
    """
    aligned_sentences = []
    
    # Ensure speaker segments are sorted by start time
    speaker_segments = sorted(speaker_segments, key=lambda x: x['start'])
    
    # For debugging - show segment boundaries
    segment_ranges = [f"{seg['start']:.2f}-{seg['end']:.2f}: {seg['speaker']}" for seg in speaker_segments[:10]]
    st.write("Speaker segment timeline (first 10):")
    st.code("\n".join(segment_ranges))
    
    # For each sentence, try to find its timing and corresponding speaker
    for i, sentence in enumerate(sentences):
        # STEP 1: Get reliable timing for this sentence
        sentence_timing = None
        
        if word_timestamps:
            sentence_words = sentence.split()
            matching_timestamps = []
            
            # Find ANY words that match between this sentence and timestamps
            for word_info in word_timestamps:
                word = word_info['word'].strip().lower()
                if any(w.lower() in word or word in w.lower() for w in sentence_words):
                    matching_timestamps.append(word_info)
            
            if matching_timestamps:
                # Sort by time and take first/last
                matching_timestamps.sort(key=lambda x: x['start'])
                sentence_timing = {
                    'start': matching_timestamps[0]['start'],
                    'end': matching_timestamps[-1]['end']
                }
        
        # STEP 2: Find which speaker segment contains this sentence
        # Display timing for debugging
        st.write(f"Sentence {i}: '{sentence[:40]}...' estimated time: {sentence_timing['start']:.2f}-{sentence_timing['end']:.2f}")
        
        # Find ALL overlapping segments
        overlapping_segments = []
        for segment in speaker_segments:
            # Check for overlap
            overlap_start = max(sentence_timing['start'], segment['start'])
            overlap_end = min(sentence_timing['end'], segment['end'])
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                overlapping_segments.append({
                    'speaker': segment['speaker'],
                    'duration': overlap_duration
                })
        
        # Choose speaker with most overlap
        if overlapping_segments:
            best_match = max(overlapping_segments, key=lambda x: x['duration'])
            speaker = best_match['speaker']
        else:
            # Find nearest segment if no overlap
            mid_sentence = (sentence_timing['start'] + sentence_timing['end']) / 2
            nearest_segment = min(speaker_segments, key=lambda x: 
                                 min(abs(x['start'] - mid_sentence), abs(x['end'] - mid_sentence)))
            speaker = nearest_segment['speaker']
        
        # Add to results
        aligned_sentences.append({
            'start': round(sentence_timing['start'], 2),
            'end': round(sentence_timing['end'], 2),
            'text': sentence,
            'speaker': speaker
        })
    
    st.write(f"... aligned {len(sentences)} sentences")
    
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
        
        # Process audio with enhanced multilanguage capability
        st.write("Processing audio...")
        audio_results = handle_multilanguage_audio(temp_file_path)
        
        # Display language information
        st.write(f"Primary Language: **{audio_results['primary_language']}**")
        
        if audio_results['multiple_languages_detected']:
            st.write("### Multiple Languages Detected")
            st.write(audio_results['language_analysis'])
        
        # Display original transcription
        st.write("### Original Transcription:")
        st.text_area("Transcript", audio_results['transcription'], height=200)
        
        # Display translation if available
        if audio_results['translation']:
            st.write("### English Translation:")
            st.text_area("Translation", audio_results['translation'], height=200)
            # If the translation exists, use it if it's in English
            if audio_results['primary_language'] == "en":
                text_for_analysis = audio_results['translation']
            else:
                text_for_analysis = audio_results['transcription']
        else:
            text_for_analysis = audio_results['transcription']
        
        # Speaker Diarization with improved function
        st.write("Performing Speaker Diarization...")
        try:
            diarization_pipeline = load_diarization_pipeline()
            speaker_segments = diarize_audio(diarization_pipeline, temp_file_path)
            
            # Align sentences with speakers
            st.write("Aligning transcription with speaker labels...")
            sentences = [seg["text"] for seg in speaker_segments]
            sentences_with_speakers = [
                {"speaker": seg["speaker"], "text": seg["text"]}
                for seg in speaker_segments
            ]

            # Sentiment Analysis
            st.write("Performing Sentiment Analysis...")
            messages = [s["text"] for s in sentences_with_speakers]
            sentiments = batch_analyze_sentiments(messages)
            
            # When preparing the final results DataFrame
            results = []
            for i, sentiment in enumerate(sentiments):
                if i < len(sentences_with_speakers):  # Safety check
                    speaker = sentences_with_speakers[i]["speaker"]
                    confidence = sentences_with_speakers[i].get("confidence", "unknown")
                    
                    # Add debugging info right in the results
                    results.append({
                        "Speaker": speaker,
                        "Confidence": confidence,
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
            st.write("Final Analysis:")
            st.dataframe(df)
            
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
            st.write("Basic Sentiment Analysis (without speaker identification):")
            st.dataframe(df)
    
        # Clean up
        os.remove(temp_file_path)
