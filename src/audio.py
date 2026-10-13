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
from itertools import groupby

import numpy as np
import soundfile as sf

from src.schemas import (
    AlignedSentence,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000

UNKNOWN_SPEAKER = "Unknown Speaker"
# Whisper and pyannote boundaries rarely line up exactly, so a word or segment
# that falls in a gap between diarized turns goes to the nearest speaker when
# that speaker is at most this many seconds away.
MAX_SPEAKER_GAP = 1.0

# MIME types for the browser audio player, one per extension the uploader accepts.
AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}


def audio_mime_type(file_name: str) -> str:
    """Return the MIME type for an uploaded audio file, by extension."""
    file_extension = os.path.splitext(file_name)[1].lower()
    return AUDIO_MIME.get(file_extension, "audio/wav")


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


def transcribe_audio(wav_path: str, translate: bool = False) -> TranscriptionResult:
    """Transcribe audio locally with faster-whisper.

    Returns a :class:`TranscriptionResult` with the text, detected language,
    per-segment and per-word timestamps, and — when ``translate`` is set and the
    audio isn't English — an English translation. Translating runs Whisper a
    second time over the whole file, so it roughly doubles the work.
    """
    from src.models import load_asr_model  # lazy: keeps module import cheap

    model = load_asr_model()

    # vad_filter skips silence and music, where Whisper otherwise tends to
    # invent text; word timestamps let speaker alignment split a segment that
    # spans a change of speaker.
    segments, info = model.transcribe(
        wav_path, beam_size=5, vad_filter=True, word_timestamps=True
    )
    transcript_segments = []
    parts = []
    for seg in segments:  # generator — consume once
        text = seg.text.strip()
        if not text:
            continue
        parts.append(text)
        words = [
            TranscriptWord(text=w.word, start=float(w.start), end=float(w.end))
            for w in (seg.words or [])
        ]
        transcript_segments.append(
            TranscriptSegment(
                text=text, start=float(seg.start), end=float(seg.end), words=words
            )
        )

    transcription = " ".join(parts).strip()
    primary_language = info.language or "unknown"

    translation = None
    if translate and primary_language not in ("en", "unknown"):
        try:
            tr_segments, _ = model.transcribe(
                wav_path, task="translate", beam_size=5, vad_filter=True
            )
            translation = " ".join(s.text.strip() for s in tr_segments).strip()
        except Exception as e:
            logger.warning("Translation failed: %s", e)

    return TranscriptionResult(
        transcription=transcription,
        language=primary_language,
        translation=translation,
        segments=transcript_segments,
    )


def diarize_audio(
    diarization_pipeline, wav_path: str, num_speakers: int | None = None
) -> list[SpeakerSegment]:
    """Run speaker diarization and return a list of :class:`SpeakerSegment`.

    The audio is loaded in-memory and passed as a waveform dict, which avoids
    pyannote 4.x's file-decoding path (torchcodec), unreliable on Windows.
    ``num_speakers``, when known, tells pyannote exactly how many speakers to
    find instead of estimating it.

    Raises ``RuntimeError`` on failure or an empty result rather than returning a
    fabricated single-speaker segment — a bad diarization should surface as an
    error, not masquerade as a valid result.
    """
    import torch  # lazy: only needed when diarization actually runs

    audio, sr = _load_waveform(wav_path)
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (channel, time)

    options = {"num_speakers": num_speakers} if num_speakers else {}
    try:
        diarization_result = diarization_pipeline(
            {"waveform": waveform, "sample_rate": sr}, **options
        )
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


def _speaker_for_span(
    start: float, end: float, speaker_segments: list[SpeakerSegment]
) -> str | None:
    """The speaker overlapping ``[start, end]`` the most.

    With no overlap (e.g. the span sits in a gap between turns, or is a
    zero-length word), falls back to the nearest speaker within
    ``MAX_SPEAKER_GAP`` seconds; returns ``None`` if there is none.
    """
    best_speaker, best_overlap = None, 0.0
    nearest_speaker, nearest_gap = None, float("inf")
    for sp in speaker_segments:
        overlap = min(end, sp.end) - max(start, sp.start)  # negative = gap
        if overlap > best_overlap:
            best_speaker, best_overlap = sp.speaker, overlap
        gap = max(-overlap, 0.0)
        if gap < nearest_gap:
            nearest_speaker, nearest_gap = sp.speaker, gap
    if best_speaker:
        return best_speaker
    return nearest_speaker if nearest_gap <= MAX_SPEAKER_GAP else None


def _split_segment_by_speaker(
    seg: TranscriptSegment, speaker_segments: list[SpeakerSegment]
) -> list[AlignedSentence]:
    """Assign each word of ``seg`` a speaker and split where the speaker changes."""
    labels = [_speaker_for_span(w.start, w.end, speaker_segments) for w in seg.words]
    first_known = next((label for label in labels if label), None)
    if first_known is None:
        return [AlignedSentence(seg.text, seg.start, seg.end, UNKNOWN_SPEAKER)]

    # A word with no speaker nearby joins the turn before it (or, at the start,
    # the first known one) rather than starting a turn of its own.
    filled, current = [], first_known
    for label in labels:
        current = label or current
        filled.append(current)

    runs = [
        (speaker, [word for word, _ in group])
        for speaker, group in groupby(zip(seg.words, filled), key=lambda pair: pair[1])
    ]
    if len(runs) == 1:  # one speaker throughout: keep Whisper's segment as-is
        return [AlignedSentence(seg.text, seg.start, seg.end, runs[0][0])]
    return [
        AlignedSentence(
            text="".join(w.text for w in words).strip(),
            start=words[0].start,
            end=words[-1].end,
            speaker=speaker,
        )
        for speaker, words in runs
    ]


def assign_speakers_to_sentences(
    transcription: TranscriptionResult, speaker_segments: list[SpeakerSegment]
) -> list[AlignedSentence]:
    """Attribute the transcript to speakers.

    Segments with word timings are split wherever the speaker changes, so a
    segment spanning two speakers becomes two rows. Segments without word
    timings go whole to the speaker they overlap most. Returns a list of
    :class:`AlignedSentence`.
    """
    segments = transcription.segments
    if not segments or not speaker_segments:
        return []

    result = []
    for seg in segments:
        if seg.words:
            result.extend(_split_segment_by_speaker(seg, speaker_segments))
        else:
            speaker = _speaker_for_span(seg.start, seg.end, speaker_segments)
            result.append(
                AlignedSentence(seg.text, seg.start, seg.end, speaker or UNKNOWN_SPEAKER)
            )
    return result
