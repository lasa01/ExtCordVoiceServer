from dataclasses import dataclass


@dataclass
class TtsRequest:
    text: str


@dataclass
class TtsResponse:
    text: str
    audio_data: bytes


@dataclass
class AsrRequest:
    opus_packets: bytes
    # keyword: str


@dataclass
class AsrResponse:
    text: str
    text_phonetic: str
    # keyword: str
    # keyword_phonetic: str
    # keyword_confidence: float
    # rest: str
    # rest_phonetic: str
