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
    Improved alignment between sentences and speaker segments with better debug info.
    """
    aligned_sentences = []
    debug_info = []
    
    # First, ensure speaker_segments are sorted by start time
    speaker_segments = sorted(speaker_segments, key=lambda x: x['start'])
    
    for i, sentence in enumerate(sentences):
        sentence_words = sentence.split()
        sentence_start = None
        sentence_end = None
        
        # Get better timing information from word timestamps
        if word_timestamps:
            # Find words that appear in the sentence
            matching_words = []
            for word_data in word_timestamps:
                word = word_data['word'].strip().lower()
                if any(word in w.lower() for w in sentence_words):
                    matching_words.append(word_data)
            
            if matching_words:
                # Sort by start time
                matching_words.sort(key=lambda x: x['start'])
                sentence_start = matching_words[0]['start']
                sentence_end = matching_words[-1]['end']
                
        # If word matching failed, estimate based on sentence position
        if sentence_start is None:
            # Estimate timing based on position in the transcript
            if i == 0:
                sentence_start = 0.0
            else:
                prev_end = aligned_sentences[-1]['end'] if aligned_sentences else 0.0
                sentence_start = prev_end
            
            # Estimate ~2 seconds per 5 words as a fallback
            words_duration = len(sentence_words) * 0.4  # 0.4 sec per word estimate
            sentence_end = sentence_start + max(words_duration, 1.0)
        
        # Add small buffers to increase chance of overlap
        search_start = max(0, sentence_start - 0.5)
        search_end = sentence_end + 0.5
        
        # Find the speaker(s) active during this sentence
        overlapping_segments = []
        for segment in speaker_segments:
            # Check for any overlap
            if segment['end'] > search_start and segment['start'] < search_end:
                overlap_start = max(search_start, segment['start'])
                overlap_end = min(search_end, segment['end'])
                overlap_duration = overlap_end - overlap_start
                
                overlapping_segments.append({
                    'speaker': segment['speaker'],
                    'overlap': overlap_duration
                })
        
        # Debug info
        segment_debug = {
            'sentence_idx': i,
            'sentence_text': sentence[:30] + ('...' if len(sentence) > 30 else ''),
            'estimated_timing': f"{sentence_start:.2f} - {sentence_end:.2f}",
            'overlapping_speakers': [f"{s['speaker']} ({s['overlap']:.2f}s)" for s in overlapping_segments]
        }
        debug_info.append(segment_debug)
        
        # Determine the primary speaker (with most overlap)
        if overlapping_segments:
            primary_speaker = max(overlapping_segments, key=lambda x: x['overlap'])['speaker']
        else:
            # No overlap found - find closest segment
            distances = []
            for segment in speaker_segments:
                # Calculate distance to the middle of the sentence
                sentence_mid = (sentence_start + sentence_end) / 2
                segment_mid = (segment['start'] + segment['end']) / 2
                distance = abs(sentence_mid - segment_mid)
                distances.append((distance, segment['speaker']))
            
            primary_speaker = min(distances, key=lambda x: x[0])[1] if distances else "Unknown"
            segment_debug['speaker_method'] = "nearest (no overlap)"
        
        aligned_sentences.append({
            "start": round(sentence_start, 2),
            "end": round(sentence_end, 2),
            "text": sentence,
            "speaker": primary_speaker,
            "confidence": "high" if overlapping_segments else "low"
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
