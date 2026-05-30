"""Typed data models exchanged between the core pipeline and the UI layer.

Using dataclasses instead of bare dicts gives the pipeline a clear, testable
contract that does not depend on Streamlit or pandas. The ``src`` package is the
"core" — it returns these objects and raises exceptions; ``app.py`` is the only
place that talks to Streamlit.
"""

from dataclasses import dataclass, field


@dataclass
class Turn:
    """One line of a pasted conversation."""

    speaker: str
    message: str


@dataclass
class SentimentResult:
    """A single classification: label + confidence."""

    sentiment: str  # POSITIVE / NEUTRAL / NEGATIVE
    confidence: float


@dataclass
class TranscriptSegment:
    """A transcribed chunk with its time span (seconds)."""

    text: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    """Full transcription output for one audio file."""

    transcription: str
    language: str
    translation: str | None
    segments: list[TranscriptSegment] = field(default_factory=list)


@dataclass
class SpeakerSegment:
    """A diarized speech span attributed to a speaker."""

    start: float
    end: float
    speaker: str  # friendly label, e.g. "Speaker 1"
    original_speaker: str | None = None  # raw pyannote label


@dataclass
class AlignedSentence:
    """A transcript segment after it has been assigned to a speaker."""

    text: str
    start: float
    end: float
    speaker: str
