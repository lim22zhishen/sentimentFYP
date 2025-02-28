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
import numpy as np

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
    """
    More robust alignment of sentences with speaker diarization results.
    
    Args:
        sentences: List of sentence strings
        word_timestamps: List of word timing information from Whisper
        speaker_segments: List of speaker segments from diarization
    
    Returns:
        List of dictionaries with aligned sentences and speakers
    """
    aligned_sentences = []
    
    # Ensure speaker segments are sorted by start time
    speaker_segments = sorted(speaker_segments, key=lambda x: x['start'])
    
    # Create a debug log
    debug_logs = []
    debug_logs.append(f"Aligning {len(sentences)} sentences with {len(speaker_segments)} speaker segments")
    
    # Map word timestamps to sentences
    sentence_word_map = map_words_to_sentences(sentences, word_timestamps)
    
    # For each sentence, determine timing and speaker
    for i, sentence in enumerate(sentences):
        sentence_obj = {"text": sentence, "alignment_method": "unknown"}
        
        # STEP 1: Get accurate timing for this sentence using multiple methods
        sentence_timing = get_sentence_timing(i, sentence, sentence_word_map, aligned_sentences)
        sentence_obj.update(sentence_timing)
        
        # STEP 2: Determine speaker using weighted methods
        speaker_info = determine_speaker(sentence_timing, speaker_segments)
        sentence_obj.update(speaker_info)
        
        # Add detailed debug information
        debug_info = (f"Sentence {i}: '{sentence[:30]}...' | "
                     f"Time: {sentence_timing['start']:.2f}-{sentence_timing['end']:.2f} | "
                     f"Speaker: {speaker_info['speaker']} | "
                     f"Confidence: {speaker_info['confidence']:.2f} | "
                     f"Method: {sentence_timing['method']}")
        debug_logs.append(debug_info)
        
        aligned_sentences.append(sentence_obj)
    
    # Display summary of debug logs
    st.write(f"Alignment completed. Debug information for first few sentences:")
    st.code("\n".join(debug_logs[:min(5, len(debug_logs))]))
    if len(debug_logs) > 5:
        st.write(f"... plus {len(debug_logs)-5} more sentences")
    
    # Validate results
    validate_alignment(aligned_sentences, speaker_segments)
    
    return aligned_sentences

def map_words_to_sentences(sentences, word_timestamps):
    """
    Map words with timestamps to sentences they belong in.
    
    Args:
        sentences: List of sentence strings
        word_timestamps: List of word timing information
    
    Returns:
        Dictionary mapping sentence index to list of word timing info
    """
    if not word_timestamps:
        return {}
    
    sentence_word_map = {i: [] for i in range(len(sentences))}
    
    # Preprocess sentences for better word matching
    processed_sentences = [preprocess_text(s) for s in sentences]
    
    # For each word with timing, find which sentence it belongs to
    for word_info in word_timestamps:
        word = preprocess_text(word_info['word'])
        
        # Skip empty words or punctuation
        if not word or len(word) <= 1:
            continue
            
        # Find which sentence contains this word
        best_match = -1
        highest_score = 0
        
        for i, sentence in enumerate(processed_sentences):
            # Skip sentences that already have many words mapped
            if len(sentence_word_map[i]) > len(sentence.split()) * 1.5:
                continue
                
            # Calculate match score (exact match, contains word, or word contains)
            if word == sentence:
                score = 1.0
            elif word in sentence:
                score = 0.9
            elif any(w == word for w in sentence.split()):
                score = 0.8
            elif any(word in w for w in sentence.split()):
                score = 0.5
            else:
                score = 0
                
            if score > highest_score:
                highest_score = score
                best_match = i
        
        # If we found a match, add this word to that sentence
        if best_match >= 0 and highest_score > 0.3:
            sentence_word_map[best_match].append(word_info)
    
    # Sort words within each sentence by start time
    for i in sentence_word_map:
        sentence_word_map[i] = sorted(sentence_word_map[i], key=lambda x: x['start'])
    
    return sentence_word_map

def preprocess_text(text):
    """Basic text preprocessing to improve word matching"""
    if not text:
        return ""
    # Convert to lowercase, remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text
    
# Highlight positive and negative sentiments
def style_table(row):
    if row["Sentiment"] == "POSITIVE":
        return ['background-color: #d4edda'] * len(row)
    elif row["Sentiment"] == "NEGATIVE":
        return ['background-color: #f8d7da'] * len(row)
    else:
        return [''] * len(row)

def get_sentence_timing(index, sentence, sentence_word_map, previous_alignments):
    """
    Get accurate timing for a sentence using multiple methods.
    
    Args:
        index: Index of the sentence
        sentence: The sentence text
        sentence_word_map: Mapping of sentences to word timestamps
        previous_alignments: Previously aligned sentences
    
    Returns:
        Dictionary with start, end time and confidence
    """
    # Method 1: Use word timestamps if available
    if index in sentence_word_map and sentence_word_map[index]:
        words = sentence_word_map[index]
        start = words[0]['start']
        end = words[-1]['end']
        
        # Calculate confidence based on number of words matched
        sentence_word_count = len(sentence.split())
        mapped_word_count = len(words)
        confidence = min(1.0, mapped_word_count / max(1, sentence_word_count))
        
        return {
            'start': start,
            'end': end,
            'timing_confidence': confidence,
            'method': 'word_timestamps'
        }
    
    # Method 2: Estimate based on previous sentence and average speaking rate
    if previous_alignments:
        prev = previous_alignments[-1]
        start = prev['end'] + 0.1  # Small gap between sentences
        
        # Estimate duration based on word count and average speaking rate
        word_count = len(sentence.split())
        duration = max(0.5, word_count * 0.3)  # Average 0.3 seconds per word
        end = start + duration
        
        return {
            'start': start,
            'end': end,
            'timing_confidence': 0.5,  # Medium confidence for this method
            'method': 'estimated_from_previous'
        }
    
    # Method 3: Fallback for first sentence without timestamps
    return {
        'start': 0.0,
        'end': max(1.0, len(sentence.split()) * 0.3),
        'timing_confidence': 0.3,  # Low confidence
        'method': 'fallback_estimate'
    }

def determine_speaker(sentence_timing, speaker_segments):
    """
    Determine which speaker most likely spoke this sentence.
    
    Args:
        sentence_timing: Dictionary with start and end time
        speaker_segments: List of speaker segments
    
    Returns:
        Dictionary with speaker and confidence
    """
    start, end = sentence_timing['start'], sentence_timing['end']
    timing_confidence = sentence_timing.get('timing_confidence', 0.5)
    
    # Find all overlapping segments
    overlapping_segments = []
    for segment in speaker_segments:
        # Calculate overlap
        overlap_start = max(start, segment['start'])
        overlap_end = min(end, segment['end'])
        
        if overlap_end > overlap_start:
            overlap_duration = overlap_end - overlap_start
            sentence_duration = end - start
            # Fraction of sentence covered by this segment
            coverage = overlap_duration / sentence_duration
            
            overlapping_segments.append({
                'speaker': segment['speaker'],
                'overlap_duration': overlap_duration,
                'coverage': coverage,
                'segment': segment
            })
    
    # If we found overlapping segments
    if overlapping_segments:
        # Sort by coverage (what fraction of the sentence this speaker covers)
        overlapping_segments.sort(key=lambda x: x['coverage'], reverse=True)
        best_match = overlapping_segments[0]
        
        # Check if we need to split the sentence (significant secondary speaker)
        needs_split = False
        if len(overlapping_segments) > 1:
            second_best = overlapping_segments[1]
            # If second speaker covers at least 30% of the sentence
            if second_best['coverage'] > 0.3:
                needs_split = True
        
        return {
            'speaker': best_match['speaker'],
            'confidence': best_match['coverage'] * timing_confidence,
            'needs_split': needs_split,
            'multiple_speakers': len(overlapping_segments) > 1
        }
    
    # If no overlap, find nearest segment
    else:
        mid_point = (start + end) / 2
        
        # Sort segments by distance to sentence midpoint
        nearest_segments = sorted(
            speaker_segments,
            key=lambda x: min(abs(x['start'] - mid_point), abs(x['end'] - mid_point))
        )
        
        if nearest_segments:
            nearest = nearest_segments[0]
            # Distance from sentence to nearest segment
            distance = min(abs(nearest['start'] - mid_point), abs(nearest['end'] - mid_point))
            # Convert distance to confidence (closer = higher confidence)
            # Confidence decays exponentially with distance
            confidence = max(0.1, min(0.5, np.exp(-distance)))
            
            return {
                'speaker': nearest['speaker'],
                'confidence': confidence * timing_confidence,
                'needs_split': False,
                'multiple_speakers': False
            }
    
    # Ultimate fallback
    return {
        'speaker': "Unknown Speaker",
        'confidence': 0.1,
        'needs_split': False,
        'multiple_speakers': False
    }

def validate_alignment(aligned_sentences, speaker_segments):
    """
    Validate alignment results and report potential issues.
    
    Args:
        aligned_sentences: List of aligned sentence objects
        speaker_segments: List of speaker segments
    """
    # Check if any sentences need to be split due to multiple speakers
    split_needed = [i for i, s in enumerate(aligned_sentences) if s.get('needs_split', False)]
    if split_needed:
        st.warning(f"{len(split_needed)} sentences might span multiple speakers and should ideally be split.")
    
    # Check for low confidence alignments
    low_confidence = [i for i, s in enumerate(aligned_sentences) if s.get('confidence', 1.0) < 0.4]
    if low_confidence:
        st.warning(f"{len(low_confidence)} sentences have low confidence speaker assignment.")
    
    # Check speaker distribution
    speaker_counts = {}
    for s in aligned_sentences:
        speaker = s.get('speaker', 'Unknown')
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    
    st.write("Speaker distribution in aligned sentences:")
    for speaker, count in speaker_counts.items():
        percentage = (count / len(aligned_sentences)) * 100
        st.write(f"- {speaker}: {count} sentences ({percentage:.1f}%)")

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
            sentences_with_speakers = align_sentences_with_diarization(
                sentences, 
                audio_results['word_timestamps'], 
                speaker_segments
            )

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
