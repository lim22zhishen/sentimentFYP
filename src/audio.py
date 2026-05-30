"""Audio processing: transcription, translation, diarization, alignment.

All inference is local: faster-whisper for transcription, language detection
and English translation, and pyannote for speaker diarization. Nothing is sent
to an external API.
"""

import os
import shutil
import subprocess

import numpy as np
import soundfile as sf
import streamlit as st
import torch

from src.models import load_asr_model, load_diarization_pipeline

TARGET_SAMPLE_RATE = 16000


def _ffmpeg_exe():
    """Return a usable ffmpeg executable path.

    Prefers a system ``ffmpeg`` on PATH; otherwise falls back to the binary
    bundled with ``imageio-ffmpeg`` so users don't have to install ffmpeg.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def process_audio_file(uploaded_file):
    """Save an uploaded file and convert it to 16 kHz mono WAV for processing."""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    temp_file_path = f"temp_audio{file_extension}"

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    if file_extension != ".wav":
        try:
            wav_path = "temp_audio.wav"
            subprocess.run(
                [_ffmpeg_exe(), "-y", "-i", temp_file_path,
                 "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1", wav_path],
                check=True, capture_output=True,
            )
            os.remove(temp_file_path)
            temp_file_path = wav_path
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to convert audio: {e.stderr.decode(errors='ignore')}")

    return temp_file_path


def _load_waveform(wav_path):
    """Load a WAV file as a mono float32 numpy array at TARGET_SAMPLE_RATE."""
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:  # stereo -> mono
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio), sr


def transcribe_audio(wav_path):
    """Transcribe audio locally with faster-whisper, returning text, detected
    language, per-segment timestamps, and an English translation when needed."""
    model = load_asr_model()

    segments, info = model.transcribe(wav_path, beam_size=5)
    sentence_timestamps = []
    parts = []
    for seg in segments:  # generator — consume once
        text = seg.text.strip()
        parts.append(text)
        sentence_timestamps.append({
            "text": text,
            "start": float(seg.start),
            "end": float(seg.end),
        })

    transcription = " ".join(parts).strip()
    primary_language = info.language or "unknown"

    # Translate to English (single pass, for display) when not already English.
    translation = None
    if primary_language not in ("en", "unknown"):
        try:
            tr_segments, _ = model.transcribe(wav_path, task="translate", beam_size=5)
            translation = " ".join(s.text.strip() for s in tr_segments).strip()
        except Exception as e:
            st.warning(f"Translation failed: {e}")

    return {
        "transcription": transcription,
        "primary_language": primary_language,
        "translation": translation,
        "sentence_timestamps": sentence_timestamps,
    }


def diarize_audio(diarization_pipeline, wav_path):
    """Run speaker diarization and return segments mapped to friendly names.

    The audio is loaded in-memory and passed as a waveform dict, which avoids
    pyannote 4.x's file-decoding path (torchcodec), unreliable on Windows.
    """
    try:
        st.write("Starting diarization...")
        audio, sr = _load_waveform(wav_path)
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (channel, time)
        diarization_result = diarization_pipeline({"waveform": waveform, "sample_rate": sr})

        # pyannote 4.x returns a DiarizeOutput wrapper; 3.x returns the
        # Annotation directly. Unwrap to the Annotation either way.
        annotation = getattr(diarization_result, "speaker_diarization", diarization_result)

        speaker_segments = []
        unique_speakers = set()
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            unique_speakers.add(speaker)
            speaker_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "speaker": speaker,
            })

        st.success(
            f"Diarization complete. Found {len(unique_speakers)} unique speakers "
            f"and {len(speaker_segments)} speech segments."
        )

        speaker_map = {s: f"Speaker {i+1}" for i, s in enumerate(sorted(unique_speakers))}
        mapped_segments = []
        for segment in speaker_segments:
            mapped = segment.copy()
            mapped["original_speaker"] = segment["speaker"]
            mapped["speaker"] = speaker_map.get(segment["speaker"], segment["speaker"])
            mapped_segments.append(mapped)

        return mapped_segments

    except Exception as e:
        st.error(f"Speaker diarization failed: {str(e)}")
        return [{"start": 0.0, "end": 1000.0, "speaker": "Speaker 1"}]


def assign_speakers_to_sentences(audio_results, speaker_segments):
    """Assign each transcribed sentence to the speaker with the largest overlap."""
    sentence_timestamps = audio_results.get("sentence_timestamps", [])
    result = []

    if not sentence_timestamps or not speaker_segments:
        return []

    for sentence_info in sentence_timestamps:
        sentence_start = sentence_info["start"]
        sentence_end = sentence_info["end"]
        assigned_speaker = "Unknown Speaker"
        best_overlap = 0

        for speaker_segment in speaker_segments:
            speaker_start = speaker_segment["start"]
            speaker_end = speaker_segment["end"]
            if sentence_start <= speaker_end and sentence_end >= speaker_start:
                overlap = min(sentence_end, speaker_end) - max(sentence_start, speaker_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    assigned_speaker = speaker_segment["speaker"]

        result.append({
            "text": sentence_info["text"],
            "start": sentence_start,
            "end": sentence_end,
            "speaker": assigned_speaker,
        })

    return result
