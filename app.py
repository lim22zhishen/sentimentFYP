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
    Perform speaker diarization using PyAnnote with improved handling of speaker labels.
    """
    try:
        diarization_result = diarization_pipeline(audio_file)
        speaker_segments = []
        speaker_mapping = {}  # Dictionary to store unique speaker labels
        speaker_index = 0  # Start indexing speakers
        
        for segment, _, speaker in diarization_result.itertracks(yield_label=True):
            if speaker not in speaker_mapping:
                speaker_mapping[speaker] = f"SPEAKER_{speaker_index}"
                speaker_index += 1  # Increment speaker count only for new speakers
            
            speaker_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "speaker": speaker_mapping[speaker]
            })
        
        if not speaker_segments:
            st.warning("No speakers detected. Check if the audio contains clear speech.")
        
        # After diarization, give option to rename speakers
        if speaker_segments:
            st.subheader("Speaker Identification")
            st.write("The following speakers were detected. You can rename them if needed:")
            
            new_mapping = {}
            for original_label in set(segment["speaker"] for segment in speaker_segments):
                default_name = original_label
                new_name = st.text_input(f"Rename {original_label}", value=default_name)
                new_mapping[original_label] = new_name
            
            # Update speaker names if the user modified them
            for segment in speaker_segments:
                segment["speaker"] = new_mapping.get(segment["speaker"], segment["speaker"])
        
        return speaker_segments

    except Exception as e:
        st.error(f"Speaker diarization failed: {str(e)}")
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
    
# Function to split transcription into sentences
def split_into_sentences(transcription):
    # Use regex to split by punctuation, keeping sentence structure
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', transcription)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def align_sentences_with_diarization(original_sentences, translated_sentences, word_timestamps, speaker_segments):
    """
    Align sentences with speaker diarization results, handling translations properly.
    
    Parameters:
    - original_sentences: List of sentences from original transcription
    - translated_sentences: List of translated sentences (or None if no translation)
    - word_timestamps: Word-level timestamps from the original transcription
    - speaker_segments: Speaker diarization segments
    
    Returns:
    - List of aligned sentences with speaker information
    """
    aligned_sentences = []
    
    # Determine which sentences to use for display/analysis
    sentences_to_align = translated_sentences if translated_sentences else original_sentences
    
    # If we have translations with different sentence counts, we need mapping
    if translated_sentences and len(translated_sentences) != len(original_sentences):
        st.warning("Translation resulted in different sentence boundaries - speaker assignment is approximate")
        ratio = len(original_sentences) / max(1, len(translated_sentences))
        
        # Now align each translated sentence
        for i, sentence in enumerate(sentences_to_align):
            # Map back to original timestamp position
            original_index = min(int(i * ratio), len(original_sentences) - 1)
            
            # Use original sentence for timestamp lookup
            words = original_sentences[original_index].split()
            sentence_start = None
            sentence_end = None
            
            # Find timestamps using the original words
            if word_timestamps:
                for word_data in word_timestamps:
                    word = word_data['word'].strip().lower()
                    if any(word in w.lower() for w in words[:2]) and sentence_start is None:
                        sentence_start = word_data['start']
                    if any(word in w.lower() for w in words[-2:]):
                        sentence_end = word_data['end']
                        break
            
            # Fallback timestamps
            sentence_start = round(sentence_start or 0.0, 2)
            sentence_end = round(sentence_end or sentence_start + 5, 2)
            
            # Find most likely speaker (maximum overlap)
            speaker = "Unknown"
            max_overlap = 0
            
            for segment in speaker_segments:
                segment_start, segment_end = segment['start'], segment['end']
                overlap_start = max(sentence_start, segment_start)
                overlap_end = min(sentence_end, segment_end)
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    if overlap_duration > max_overlap:
                        max_overlap = overlap_duration
                        speaker = segment['speaker']
            
            aligned_sentences.append({
                "start": sentence_start,
                "end": sentence_end,
                "text": sentence,
                "speaker": speaker,
                "confidence": max_overlap / (sentence_end - sentence_start)  # Add confidence score
            })
    else:
        # Simple 1:1 alignment (no translation or same sentence count)
        for i, sentence in enumerate(sentences_to_align):
            # Use matching original sentence for timestamps
            words = original_sentences[i].split()
            sentence_start = None
            sentence_end = None
            
            # Find timestamps
            if word_timestamps:
                for word_data in word_timestamps:
                    word = word_data['word'].strip().lower()
                    if any(word in w.lower() for w in words[:2]) and sentence_start is None:
                        sentence_start = word_data['start']
                    if any(word in w.lower() for w in words[-2:]):
                        sentence_end = word_data['end']
                        break
            
            # Fallback
            sentence_start = round(sentence_start or 0.0, 2)
            sentence_end = round(sentence_end or sentence_start + 5, 2)
            
            # Assign speaker
            speaker = "Unknown"
            max_overlap = 0
            
            for segment in speaker_segments:
                segment_start, segment_end = segment['start'], segment['end']
                overlap_start = max(sentence_start, segment_start)
                overlap_end = min(sentence_end, segment_end)
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    if overlap_duration > max_overlap:
                        max_overlap = overlap_duration
                        speaker = segment['speaker']
            
            aligned_sentences.append({
                "start": sentence_start,
                "end": sentence_end,
                "text": sentence,
                "speaker": speaker,
                "confidence": max_overlap / max(0.1, (sentence_end - sentence_start))  # Add confidence score
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
        original_sentences = split_into_sentences(audio_results['transcription'])
        translated_sentences = None
        
        if audio_results['translation']:
            st.write("### English Translation:")
            st.text_area("Translation", audio_results['translation'], height=200)
            translated_sentences = split_into_sentences(audio_results['translation'])
            text_for_analysis = audio_results['translation']
        else:
            text_for_analysis = audio_results['transcription']
        
        # Speaker Diarization with improved function
        st.write("Performing Speaker Diarization...")
        try:
            diarization_pipeline = load_diarization_pipeline()
            speaker_segments = diarize_audio(diarization_pipeline, temp_file_path)
            
            # Allow users to rename speakers
            st.subheader("Speaker Identification")
            st.write("The following speakers were detected. You can rename them if needed:")
            
            unique_speakers = list(set(segment["speaker"] for segment in speaker_segments))
            new_mapping = {}
            
            for original_label in unique_speakers:
                default_name = original_label
                new_name = st.text_input(f"Rename {original_label}", value=default_name)
                new_mapping[original_label] = new_name
            
            # Update speaker names if the user modified them
            for segment in speaker_segments:
                segment["speaker"] = new_mapping.get(segment["speaker"], segment["speaker"])
            
            # Align sentences with speakers, handling translation properly
            st.write("Aligning transcription with speaker labels...")
            
            sentences_with_speakers = align_sentences_with_diarization(
                original_sentences,
                translated_sentences,
                audio_results['word_timestamps'],
                speaker_segments
            )
            
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
