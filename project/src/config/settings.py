from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core.paths import PathRegistry, DEFAULT_ROOT


@dataclass(slots=True)
class AudioSettings:
    sample_rate: int = 16_000
    channels: int = 1
    vad_threshold: float = 0.45
    vad_min_silence_ms: int = 1200
    vad_speech_pad_ms: int = 400
    vad_min_speech_ms: int = 250
    silence_rms_threshold: float = 0.0125
    chunk_seconds: float = 1.0


@dataclass(slots=True)
class SpeechSettings:
    whisper_model: str = "base.en"
    language_tool_server: Optional[str] = None
    normalization_locale: str = "en_US"
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_voice_clone_reference: Optional[Path] = None
    # Output sample rate for TTS/voice mimic; defaults to match the audio pipeline.
    tts_sample_rate: int = 16_000


@dataclass(slots=True)
class LLMSettings:
    enabled: bool = True
    backend: str = "ollama"
    model: str = "deepseek-r1:8b"
    max_tokens: int = 128
    temperature_base: float = 1.5
    top_p_base: float = 0.85
    top_p_spread: float = 0.15


@dataclass(slots=True)
class BehaviorSettings:
    correction_echo_enabled: bool = True
    caregiver_prompts_enabled: bool = True
    support_voice_enabled: bool = False
    max_phrase_history: int = 5
    anxious_threshold: int = 3
    perseveration_threshold: int = 3
    high_energy_rms: float = 0.08


@dataclass(slots=True)
class HeartSettings:
    num_nodes: int = 1024
    num_channels: int = 5
    dt: float = 0.03
    beta_decay: float = 0.5
    gamma_diffusion: float = 0.3
    noise_scale: float = 0.1
    anneal_k: float = 0.01
    max_abs: float = 10.0
    target_sr: int = 16_000
    arousal_gain: float = 25.0
    max_arousal: float = 10.0
    use_llm: bool = True
    llm_backend: str = "ollama"
    llm_model: str = "deepseek-r1:8b"
    llm_temperature_scale: float = 1.5
    llm_top_p_base: float = 0.9
    llm_top_p_spread: float = 0.1
    llm_max_tokens: int = 128
    embedding_dim: int = 1024
    embedding_channel: int = 4
    embedding_gain: float = 0.05
    device: str = "cpu"


@dataclass(slots=True)
class SystemSettings:
    child_id: str = "child_001"
    child_name: str = "Jackson"
    device: str = "cpu"
    audio: AudioSettings = field(default_factory=AudioSettings)
    speech: SpeechSettings = field(default_factory=SpeechSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    behavior: BehaviorSettings = field(default_factory=BehaviorSettings)
    paths: PathRegistry = field(default_factory=PathRegistry)
    heart: HeartSettings = field(default_factory=HeartSettings)

    @property
    def voice_sample(self) -> Path:
        return self.paths.voices_dir / "child_voice.wav"


def load_settings(config_path: Optional[Path] = None) -> SystemSettings:
    """
    Load settings from disk if available, otherwise fall back to defaults.
    """

    if config_path is None:
        config_path = DEFAULT_ROOT / "config.json"

    settings = SystemSettings()
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        _apply_json(settings, data)

    settings.paths.ensure_logs()
    return settings


def _apply_json(settings: SystemSettings, data: dict) -> None:
    for key, value in data.items():
        if key == "audio" and isinstance(value, dict):
            _update_dataclass(settings.audio, value)
        elif key == "speech" and isinstance(value, dict):
            _update_dataclass(settings.speech, value)
        elif key == "llm" and isinstance(value, dict):
            _update_dataclass(settings.llm, value)
        elif key == "behavior" and isinstance(value, dict):
            _update_dataclass(settings.behavior, value)
        elif key == "heart" and isinstance(value, dict):
            _update_dataclass(settings.heart, value)
        elif hasattr(settings, key):
            setattr(settings, key, value)


def _update_dataclass(obj, values: dict) -> None:
    for key, value in values.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
