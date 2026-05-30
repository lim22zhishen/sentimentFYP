"""Unit tests for src.audio pure logic + preprocessing + diarization handling."""

import os

import numpy as np
import soundfile as sf
import pytest

from src.audio import (
    assign_speakers_to_sentences,
    process_audio_file,
    _load_waveform,
    diarize_audio,
)
from src.schemas import SpeakerSegment, TranscriptionResult, TranscriptSegment


def _write_wav(path, sr=16000, seconds=0.1, channels=1):
    n = int(sr * seconds)
    if channels > 1:
        data = np.zeros((n, channels), dtype="float32")
    else:
        data = np.zeros(n, dtype="float32")
    sf.write(str(path), data, sr)


def _transcription(*segments):
    return TranscriptionResult(
        transcription="", language="en", translation=None, segments=list(segments)
    )


# --- assign_speakers_to_sentences -----------------------------------------

def test_assign_speakers_largest_overlap():
    transcription = _transcription(
        TranscriptSegment("hello", 0.0, 2.0),
        TranscriptSegment("world", 5.0, 6.0),
    )
    segments = [
        SpeakerSegment(0.0, 1.8, "Speaker 1"),
        SpeakerSegment(1.8, 4.0, "Speaker 2"),
        SpeakerSegment(4.5, 7.0, "Speaker 2"),
    ]
    out = assign_speakers_to_sentences(transcription, segments)
    assert [o.speaker for o in out] == ["Speaker 1", "Speaker 2"]
    assert out[0].text == "hello"


def test_assign_speakers_no_overlap_is_unknown():
    transcription = _transcription(TranscriptSegment("x", 10.0, 11.0))
    segments = [SpeakerSegment(0.0, 1.0, "Speaker 1")]
    out = assign_speakers_to_sentences(transcription, segments)
    assert out[0].speaker == "Unknown Speaker"


def test_assign_speakers_empty_inputs():
    assert assign_speakers_to_sentences(_transcription(), []) == []
    assert assign_speakers_to_sentences(
        _transcription(TranscriptSegment("a", 0.0, 1.0)), []
    ) == []


# --- preprocessing ----------------------------------------------------------

def test_load_waveform_mono(tmp_path):
    p = tmp_path / "m.wav"
    _write_wav(p, channels=1)
    audio, sr = _load_waveform(str(p))
    assert sr == 16000
    assert audio.ndim == 1


def test_load_waveform_stereo_is_downmixed(tmp_path):
    p = tmp_path / "s.wav"
    _write_wav(p, channels=2)
    audio, _ = _load_waveform(str(p))
    assert audio.ndim == 1


def test_process_audio_file_wav_passthrough_is_unique(tmp_path):
    src = tmp_path / "in.wav"
    _write_wav(src)
    raw = src.read_bytes()

    class FakeUpload:
        name = "in.wav"

        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

    out = process_audio_file(FakeUpload(raw))
    try:
        assert os.path.exists(out)
        assert out.endswith(".wav")
        # unique temp name, not the old fixed "temp_audio.wav"
        assert os.path.basename(out) != "temp_audio.wav"
    finally:
        if os.path.exists(out):
            os.remove(out)


# --- diarization error handling (no fabricated fallback) --------------------

class _Seg:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _Annotation:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=True):
        for seg, label in self._tracks:
            yield seg, None, label


class _Output:
    def __init__(self, annotation):
        self.speaker_diarization = annotation


def test_diarize_audio_maps_speakers(tmp_path):
    p = tmp_path / "d.wav"
    _write_wav(p)
    annotation = _Annotation([(_Seg(0.0, 1.0), "spkB"), (_Seg(1.0, 2.0), "spkA")])
    out = diarize_audio(lambda payload: _Output(annotation), str(p))
    names = {s.original_speaker: s.speaker for s in out}
    assert names == {"spkA": "Speaker 1", "spkB": "Speaker 2"}


def test_diarize_audio_raises_on_pipeline_failure(tmp_path):
    p = tmp_path / "d.wav"
    _write_wav(p)

    def boom(payload):
        raise ValueError("model exploded")

    with pytest.raises(RuntimeError, match="Speaker diarization failed"):
        diarize_audio(boom, str(p))


def test_diarize_audio_raises_on_empty_result(tmp_path):
    p = tmp_path / "d.wav"
    _write_wav(p)
    with pytest.raises(RuntimeError, match="no speech segments"):
        diarize_audio(lambda payload: _Output(_Annotation([])), str(p))
