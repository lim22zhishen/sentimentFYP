"""Audio processing: transcription, translation, diarization, alignment.

Pure core logic — returns dataclasses and raises on error, no Streamlit. All
inference is local: faster-whisper for transcription, language detection and
English translation, and pyannote for speaker diarization. Nothing is sent to an
external API.
"""

import logging
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from src.schemas import (
    AlignedSentence,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def _ffmpeg_exe() -> str:
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


def _read_upload_bytes(uploaded_file) -> bytes:
    """Read all bytes from an upload, independent of its read cursor.

    Streamlit's ``UploadedFile`` has a stateful cursor, so ``.read()`` can return
    empty if the buffer was already consumed; ``.getvalue()`` always returns the
    full content. Fall back to ``.read()`` for plain file-likes.
    """
    getvalue = getattr(uploaded_file, "getvalue", None)
    if callable(getvalue):
        return getvalue()
    return uploaded_file.read()


def _needs_conversion(path: str, file_extension: str) -> bool:
    """True unless ``path`` is already a 16 kHz mono WAV (so it can pass through)."""
    if file_extension != ".wav":
        return True
    try:
        info = sf.info(path)
        return not (info.samplerate == TARGET_SAMPLE_RATE and info.channels == 1)
    except Exception:
        return True  # unreadable header -> let ffmpeg normalize it


def _convert_to_wav(src_path: str) -> str:
    """Convert any audio file to a unique 16 kHz mono WAV via ffmpeg."""
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="sent_audio_")
    os.close(wav_fd)
    try:
        subprocess.run(
            [_ffmpeg_exe(), "-y", "-i", src_path,
             "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1", wav_path],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        raise RuntimeError(
            f"Failed to convert audio: {e.stderr.decode(errors='ignore')}"
        ) from e
    return wav_path


def process_audio_file(uploaded_file) -> str:
    """Save an upload and ensure it's a 16 kHz mono WAV for processing.

    Every input is normalized to 16 kHz mono so the models see consistent audio;
    a WAV that is already 16 kHz mono is passed through without re-encoding. Uses
    unique temp paths (``tempfile``) so two overlapping runs can't clobber each
    other. Returns the path to a WAV the caller is responsible for deleting.
    """
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    fd, src_path = tempfile.mkstemp(suffix=file_extension or ".bin", prefix="sent_audio_")
    with os.fdopen(fd, "wb") as f:
        f.write(_read_upload_bytes(uploaded_file))

    if not _needs_conversion(src_path, file_extension):
        return src_path

    try:
        return _convert_to_wav(src_path)
    finally:
        if os.path.exists(src_path):
            os.remove(src_path)


def _load_waveform(wav_path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as a mono float32 numpy array at its native sample rate."""
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:  # stereo -> mono
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio), sr


def transcribe_audio(wav_path: str) -> TranscriptionResult:
    """Transcribe audio locally with faster-whisper.

    Returns a :class:`TranscriptionResult` with the text, detected language,
    per-segment timestamps, and an English translation when needed.
    """
    from src.models import load_asr_model  # lazy: keeps module import cheap

    model = load_asr_model()

    segments, info = model.transcribe(wav_path, beam_size=5)
    transcript_segments = []
    parts = []
    for seg in segments:  # generator — consume once
        text = seg.text.strip()
        parts.append(text)
        transcript_segments.append(
            TranscriptSegment(text=text, start=float(seg.start), end=float(seg.end))
        )

    transcription = " ".join(parts).strip()
    primary_language = info.language or "unknown"

    # Translate to English (single pass, for display) when not already English.
    translation = None
    if primary_language not in ("en", "unknown"):
        try:
            tr_segments, _ = model.transcribe(wav_path, task="translate", beam_size=5)
            translation = " ".join(s.text.strip() for s in tr_segments).strip()
        except Exception as e:
            logger.warning("Translation failed: %s", e)

    return TranscriptionResult(
        transcription=transcription,
        language=primary_language,
        translation=translation,
        segments=transcript_segments,
    )


def diarize_audio(diarization_pipeline, wav_path: str) -> list[SpeakerSegment]:
    """Run speaker diarization and return a list of :class:`SpeakerSegment`.

    The audio is loaded in-memory and passed as a waveform dict, which avoids
    pyannote 4.x's file-decoding path (torchcodec), unreliable on Windows.

    Raises ``RuntimeError`` on failure or an empty result rather than returning a
    fabricated single-speaker segment — a bad diarization should surface as an
    error, not masquerade as a valid result.
    """
    import torch  # lazy: only needed when diarization actually runs

    audio, sr = _load_waveform(wav_path)
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (channel, time)

    try:
        diarization_result = diarization_pipeline({"waveform": waveform, "sample_rate": sr})
    except Exception as e:
        raise RuntimeError(f"Speaker diarization failed: {e}") from e

    # pyannote 4.x returns a DiarizeOutput wrapper; 3.x returns the Annotation
    # directly. Unwrap to the Annotation either way.
    annotation = getattr(diarization_result, "speaker_diarization", diarization_result)

    raw_segments = []
    unique_speakers = set()
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        unique_speakers.add(speaker)
        raw_segments.append((round(segment.start, 2), round(segment.end, 2), speaker))

    if not raw_segments:
        raise RuntimeError("Speaker diarization produced no speech segments.")

    speaker_map = {s: f"Speaker {i+1}" for i, s in enumerate(sorted(unique_speakers))}
    return [
        SpeakerSegment(
            start=start,
            end=end,
            speaker=speaker_map.get(spk, spk),
            original_speaker=spk,
        )
        for start, end, spk in raw_segments
    ]


def assign_speakers_to_sentences(
    transcription: TranscriptionResult, speaker_segments: list[SpeakerSegment]
) -> list[AlignedSentence]:
    """Assign each transcript segment to the speaker with the largest overlap.

    Returns a list of :class:`AlignedSentence`.
    """
    segments = transcription.segments
    if not segments or not speaker_segments:
        return []

    result = []
    for seg in segments:
        assigned_speaker = "Unknown Speaker"
        best_overlap = 0

        for sp in speaker_segments:
            if seg.start <= sp.end and seg.end >= sp.start:
                overlap = min(seg.end, sp.end) - max(seg.start, sp.start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    assigned_speaker = sp.speaker

        result.append(
            AlignedSentence(
                text=seg.text, start=seg.start, end=seg.end, speaker=assigned_speaker
            )
        )

    return result
