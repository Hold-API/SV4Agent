from pydantic import BaseModel


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: float


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[Word] | None = None


class Transcription(BaseModel):
    text: str
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    segments: list[Segment]
