from dataclasses import dataclass


@dataclass
class TtsRequest:
    language: str
    text: str


@dataclass
class TtsResponse:
    text: str
    audio_data: bytes
