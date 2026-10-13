"""Unit tests for src.audio pure logic + preprocessing + diarization handling."""

import os

import numpy as np
import soundfile as sf
import pytest

import src.models
from src.audio import (
    assign_speakers_to_sentences,
    audio_mime_type,
    process_audio_file,
    transcribe_audio,
    _load_waveform,
    diarize_audio,
)
from src.schemas import (
    SpeakerSegment,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)


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


def test_assign_speakers_segment_in_small_gap_goes_to_nearest():
    # 0.5 s after Speaker 1 stops, 2 s before Speaker 2 starts
    transcription = _transcription(TranscriptSegment("um", 1.5, 1.8))
    segments = [SpeakerSegment(0.0, 1.0, "Speaker 1"), SpeakerSegment(3.8, 5.0, "Speaker 2")]
    assert assign_speakers_to_sentences(transcription, segments)[0].speaker == "Speaker 1"


def _words(*spec):
    return [TranscriptWord(text, start, end) for text, start, end in spec]


def test_assign_speakers_splits_segment_at_speaker_change():
    seg = TranscriptSegment("Do you agree? Yes, I do.", 0.0, 3.0, words=_words(
        (" Do", 0.0, 0.3), (" you", 0.3, 0.6), (" agree?", 0.6, 1.4),
        (" Yes,", 1.9, 2.2), (" I", 2.2, 2.4), (" do.", 2.4, 3.0),
    ))
    segments = [SpeakerSegment(0.0, 1.5, "Speaker 1"), SpeakerSegment(1.8, 3.2, "Speaker 2")]
    out = assign_speakers_to_sentences(_transcription(seg), segments)
    assert [(o.speaker, o.text, o.start, o.end) for o in out] == [
        ("Speaker 1", "Do you agree?", 0.0, 1.4),
        ("Speaker 2", "Yes, I do.", 1.9, 3.0),
    ]


def test_assign_speakers_single_speaker_keeps_segment_text():
    seg = TranscriptSegment("Hello there.", 0.0, 1.0, words=_words(
        (" Hello", 0.0, 0.5), (" there.", 0.5, 1.0),
    ))
    out = assign_speakers_to_sentences(
        _transcription(seg), [SpeakerSegment(0.0, 2.0, "Speaker 1")]
    )
    assert [(o.speaker, o.text, o.start, o.end) for o in out] == [
        ("Speaker 1", "Hello there.", 0.0, 1.0)
    ]


def test_assign_speakers_unplaced_word_joins_neighbouring_turn():
    # " so" is 3 s from any diarized speech: it stays in Speaker 1's turn
    # instead of becoming an "Unknown Speaker" row of its own.
    seg = TranscriptSegment("Right, so okay", 0.0, 9.0, words=_words(
        (" Right,", 0.0, 0.5), (" so", 4.0, 4.5), (" okay", 8.5, 9.0),
    ))
    segments = [SpeakerSegment(0.0, 1.0, "Speaker 1"), SpeakerSegment(8.0, 9.0, "Speaker 2")]
    out = assign_speakers_to_sentences(_transcription(seg), segments)
    assert [(o.speaker, o.text) for o in out] == [
        ("Speaker 1", "Right, so"),
        ("Speaker 2", "okay"),
    ]


def test_assign_speakers_zero_length_word_inside_turn():
    seg = TranscriptSegment("a b", 0.0, 1.0, words=_words((" a", 0.5, 0.5), (" b", 0.5, 1.0)))
    out = assign_speakers_to_sentences(
        _transcription(seg), [SpeakerSegment(0.0, 1.0, "Speaker 1")]
    )
    assert [o.speaker for o in out] == ["Speaker 1"]


def test_assign_speakers_words_far_from_any_speaker_are_unknown():
    seg = TranscriptSegment("far away", 20.0, 21.0, words=_words(
        (" far", 20.0, 20.5), (" away", 20.5, 21.0),
    ))
    out = assign_speakers_to_sentences(
        _transcription(seg), [SpeakerSegment(0.0, 1.0, "Speaker 1")]
    )
    assert [(o.speaker, o.text) for o in out] == [("Unknown Speaker", "far away")]


# --- audio_mime_type ----------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("a.wav", "audio/wav"),
    ("a.mp3", "audio/mpeg"),
    ("a.ogg", "audio/ogg"),
    ("a.m4a", "audio/mp4"),
    ("a.flac", "audio/flac"),
    ("A.FLAC", "audio/flac"),
    ("noext", "audio/wav"),
])
def test_audio_mime_type(name, expected):
    assert audio_mime_type(name) == expected


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


class _Upload:
    """Stand-in for Streamlit's UploadedFile (exposes getvalue + read)."""

    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data

    def read(self):
        return self._data


class _ReadOnlyUpload:
    """A plain file-like with only read() — exercises the getvalue() fallback."""

    def __init__(self, name, data):
        self.name = name
        self._data = data

    def read(self):
        return self._data


def test_process_audio_file_passthrough_16k_mono(tmp_path):
    src = tmp_path / "in.wav"
    _write_wav(src, sr=16000, channels=1)

    out = process_audio_file(_Upload("in.wav", src.read_bytes()))
    try:
        assert os.path.exists(out) and out.endswith(".wav")
        # unique temp name, not the old fixed "temp_audio.wav"
        assert os.path.basename(out) != "temp_audio.wav"
        info = sf.info(out)
        assert info.samplerate == 16000 and info.channels == 1
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_process_audio_file_normalizes_stereo_44k(tmp_path):
    src = tmp_path / "in.wav"
    _write_wav(src, sr=44100, channels=2, seconds=0.2)

    out = process_audio_file(_Upload("in.wav", src.read_bytes()))
    try:
        info = sf.info(out)
        assert info.samplerate == 16000  # resampled
        assert info.channels == 1  # downmixed to mono
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_process_audio_file_falls_back_to_read(tmp_path):
    src = tmp_path / "in.wav"
    _write_wav(src)

    out = process_audio_file(_ReadOnlyUpload("in.wav", src.read_bytes()))
    try:
        assert os.path.exists(out) and out.endswith(".wav")
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


@pytest.mark.torch
def test_diarize_audio_maps_speakers(tmp_path):
    p = tmp_path / "d.wav"
    _write_wav(p)
    annotation = _Annotation([(_Seg(0.0, 1.0), "spkB"), (_Seg(1.0, 2.0), "spkA")])
    out = diarize_audio(lambda payload: _Output(annotation), str(p))
    names = {s.original_speaker: s.speaker for s in out}
    assert names == {"spkA": "Speaker 1", "spkB": "Speaker 2"}


@pytest.mark.torch
def test_diarize_audio_passes_speaker_count_only_when_given(tmp_path):
    p = tmp_path / "d.wav"
    _write_wav(p)
    annotation = _Annotation([(_Seg(0.0, 1.0), "spkA")])
    calls = []

    def pipeline(payload, **kwargs):
        calls.append(kwargs)
        return _Output(annotation)

    diarize_audio(pipeline, str(p))
    diarize_audio(pipeline, str(p), num_speakers=2)
    assert calls == [{}, {"num_speakers": 2}]


@pytest.mark.torch
def test_diarize_audio_raises_on_pipeline_failure(tmp_path):
    p = tmp_path / "d.wav"
    _write_wav(p)

    def boom(payload):
        raise ValueError("model exploded")

    with pytest.raises(RuntimeError, match="Speaker diarization failed"):
        diarize_audio(boom, str(p))


@pytest.mark.torch
def test_diarize_audio_raises_on_empty_result(tmp_path):
    p = tmp_path / "d.wav"
    _write_wav(p)
    with pytest.raises(RuntimeError, match="no speech segments"):
        diarize_audio(lambda payload: _Output(_Annotation([])), str(p))


# --- transcribe_audio (Whisper model stubbed) -------------------------------

class _FakeWord:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _FakeSegment:
    def __init__(self, text, start, end, words=None):
        self.text, self.start, self.end, self.words = text, start, end, words


class _FakeInfo:
    def __init__(self, language):
        self.language = language


class _FakeWhisper:
    def __init__(self, language="fr"):
        self.language = language
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("task") == "translate":
            return iter([_FakeSegment(" Hello.", 0.0, 1.0)]), _FakeInfo(self.language)
        segments = [
            _FakeSegment(" Bonjour.", 0.0, 1.0, [_FakeWord(" Bonjour.", 0.0, 1.0)]),
            _FakeSegment("  ", 1.0, 2.0, []),  # blank: dropped
        ]
        return iter(segments), _FakeInfo(self.language)


def test_transcribe_audio_uses_vad_and_word_timestamps(monkeypatch):
    model = _FakeWhisper()
    monkeypatch.setattr(src.models, "load_asr_model", lambda: model)
    result = transcribe_audio("x.wav")

    assert model.calls[0]["vad_filter"] is True
    assert model.calls[0]["word_timestamps"] is True
    assert result.transcription == "Bonjour."
    assert result.language == "fr"
    assert result.segments == [
        TranscriptSegment("Bonjour.", 0.0, 1.0, [TranscriptWord(" Bonjour.", 0.0, 1.0)])
    ]


def test_transcribe_audio_translates_only_when_asked(monkeypatch):
    model = _FakeWhisper()
    monkeypatch.setattr(src.models, "load_asr_model", lambda: model)

    assert transcribe_audio("x.wav").translation is None
    assert len(model.calls) == 1  # no second Whisper pass

    assert transcribe_audio("x.wav", translate=True).translation == "Hello."
    assert model.calls[-1]["task"] == "translate"


def test_transcribe_audio_skips_translation_for_english(monkeypatch):
    model = _FakeWhisper(language="en")
    monkeypatch.setattr(src.models, "load_asr_model", lambda: model)
    assert transcribe_audio("x.wav", translate=True).translation is None
    assert len(model.calls) == 1
