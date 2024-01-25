from pydantic.dataclasses import dataclass
from pydantic import TypeAdapter
from dataclasses import field
from typing import Dict, List
import json


@dataclass
class TtsLanguageConfig:
    model_path: str


@dataclass
class TtsConfig:
    languages: Dict[str, TtsLanguageConfig] = field(default_factory=dict)
    venv: str = ""


@dataclass
class AsrLanguageConfig:
    model_path: str
    accurate_model_path: str


@dataclass
class AsrConfig:
    languages: Dict[str, AsrLanguageConfig] = field(default_factory=dict)
    venv: str = ""


@dataclass
class PhoneticsConfig:
    languages: List[str] = field(default_factory=list)


@dataclass
class Config:
    tts: TtsConfig = TtsConfig()
    asr: AsrConfig = AsrConfig()
    phonetics: PhoneticsConfig = PhoneticsConfig()
    token: str = ""


ConfigValidator = TypeAdapter(Config)


def load_config_from_file(filename: str) -> Config:
    config = Config()

    try:
        obj = json.load(open(filename, encoding="utf8"))
        config = ConfigValidator.validate_python(obj)
    except FileNotFoundError:
        pass

    return config


def save_config_to_file(filename: str, config: Config):
    obj = ConfigValidator.dump_python(config)
    json.dump(obj, open(filename, "w", encoding="utf8"), indent=4)
