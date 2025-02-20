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
    try:
        # Enhanced debugging
        st.write("Starting diarization...")
        
        # Check audio file validity
        import soundfile as sf
        try:
            data, samplerate = sf.read(audio_file)
            audio_duration = len(data) / samplerate
            st.write(f"Audio file loaded successfully. Duration: {audio_duration:.2f} seconds")
            if audio_duration < 1.0:
                st.warning("Audio file is very short, which may affect diarization quality")
        except Exception as audio_error:
            st.warning(f"Audio file check warning: {str(audio_error)}")
        
        # Run diarization with progress indicator
        with st.spinner("Running speaker diarization..."):
            diarization_result = diarization_pipeline(audio_file)
        
        # Extract and validate segments
        speaker_segments = []
        speakers = set()
        
        for segment, _, speaker in diarization_result.itertracks(yield_label=True):
            speaker_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "speaker": speaker
            })
            speakers.add(speaker)
        
        # Validation
        if not speaker_segments:
            st.warning("No speaker segments detected. This might indicate an issue with the audio.")
            # Create a default segment covering the entire audio
            speaker_segments = [{"start": 0.0, "end": audio_duration, "speaker": "SPEAKER_0"}]
        
        # Merge very short segments (less than 0.5s) with adjacent segments from the same speaker
        if len(speaker_segments) > 1:
            merged_segments = [speaker_segments[0]]
            for segment in speaker_segments[1:]:
                prev = merged_segments[-1]
                # If same speaker and close together, merge
                if (segment['speaker'] == prev['speaker'] and 
                    segment['start'] - prev['end'] < 0.3):
                    prev['end'] = segment['end']
                # If very short segment, try to merge with prev
                elif segment['end'] - segment['start'] < 0.5:
                    # Keep the previous segment as is
                    pass
                else:
                    merged_segments.append(segment)
            
            speaker_segments = merged_segments
            
        st.success(f"Diarization complete. Found {len(speakers)} unique speakers and {len(speaker_segments)} speech segments.")
        return speaker_segments
        
    except Exception as e:
        st.error(f"Speaker diarization failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        # Return a default single speaker
        return [{"start": 0.0, "end": 1000.0, "speaker": "SPEAKER_0"}]

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
            
def align_sentences_with_diarization(sentences, word_timestamps, speaker_segments):
    aligned_sentences = []
    
    # Sort speaker segments by start time for efficient searching
    speaker_segments = sorted(speaker_segments, key=lambda x: x['start'])
    
    for sentence in sentences:
        # Find more reliable timestamp matches
        words = sentence.split()
        first_word_matches = [w for w in word_timestamps if w['word'].strip().lower() in words[0].lower()]
        last_word_matches = [w for w in word_timestamps if w['word'].strip().lower() in words[-1].lower()]
        
        if first_word_matches and last_word_matches:
            sentence_start = first_word_matches[0]['start']
            sentence_end = last_word_matches[-1]['end']
        else:
            # Improved fallback - look for any words in the sentence
            matching_words = [w for w in word_timestamps if any(w['word'].strip().lower() in word.lower() for word in words)]
            if matching_words:
                sentence_start = matching_words[0]['start']
                sentence_end = matching_words[-1]['end']
            else:
                # Last resort fallback
                sentence_start, sentence_end = 0.0, 5.0
        
        # Use weighted voting to determine the speaker (based on overlap duration)
        speaker_votes = {}
        for segment in speaker_segments:
            # Calculate overlap
            overlap_start = max(sentence_start, segment['start'])
            overlap_end = min(sentence_end, segment['end'])
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                speaker_votes[segment['speaker']] = speaker_votes.get(segment['speaker'], 0) + overlap_duration
        
        if speaker_votes:
            speaker = max(speaker_votes.items(), key=lambda x: x[1])[0]
        else:
            # Find the closest speaker segment if no overlap
            distances = [(abs(segment['start'] - sentence_start), segment['speaker']) for segment in speaker_segments]
            speaker = min(distances, key=lambda x: x[0])[1] if distances else "Unknown"
        
        aligned_sentences.append({
            "start": round(sentence_start, 2),
            "end": round(sentence_end, 2),
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
            sentences = split_into_sentences(text_for_analysis, audio_results['word_timestamps'])
            sentences_with_speakers = align_sentences_with_diarization(sentences, audio_results['word_timestamps'], speaker_segments)
            
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
