from pathlib import Path
from typing import Optional, Dict

import requests

from config import OPENVOICE_BASE_URL


def upload_reference_audio(wav_path: Path, voice_label: str) -> None:
    """Register a reference audio clip with OpenVoice_server."""
    url = f"{OPENVOICE_BASE_URL}/upload_audio/"
    files = {"file": wav_path.open("rb")}
    data = {"audio_file_label": voice_label}
    print(f"[OpenVoice] Uploading reference {wav_path} as '{voice_label}'...")
    resp = requests.post(url, data=data, files=files, timeout=120)
    resp.raise_for_status()
    print(f"[OpenVoice] Upload response: {resp.json()}")


def synthesize_text_with_clone(
    text: str,
    voice_label: str,
    accent: str = "en-newest",
    speed: float = 1.0,
    watermark: str = " @Bubble",
    out_path: Optional[Path] = None,
) -> bytes:
    """
    Call /synthesize_speech/ on OpenVoice_server to get cloned audio bytes.

    OpenVoice_server README defines /synthesize_speech/ like:
      GET text, voice, accent, speed, watermark -> WAV bytes.
    """
    url = f"{OPENVOICE_BASE_URL}/synthesize_speech/"
    params: Dict[str, str] = {
        "text": text,
        "voice": voice_label,
        "accent": accent,
        "speed": str(speed),
        "watermark": watermark,
    }
    print(f"[OpenVoice] Synthesizing: '{text}' as voice='{voice_label}'...")
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    audio_bytes = resp.content
    if out_path is not None:
        out_path.write_bytes(audio_bytes)
        print(f"[OpenVoice] Wrote cloned audio to {out_path}")
    return audio_bytes
