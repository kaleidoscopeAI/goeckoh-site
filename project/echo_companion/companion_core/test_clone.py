#!/usr/bin/env python3
"""
Standalone clone test harness for Jackson's Companion.

Usage (from project/companion_core):

    source .venv/bin/activate
    python test_clone.py

This will:
  - Use the same CONFIG / PathsConfig as the main loop.
  - Instantiate VoiceCrystal with the configured speaker_ref_dir.
  - Synthesize a few fixed first-person sentences.
  - Print per-utterance latency and basic stats.
  - Concatenate them into clone_test.wav in the current directory.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import List

import numpy as np

from jackson_companion_v15_3 import (
    CONFIG,
    VoiceCrystal,
    VoiceCrystalConfig,
    enforce_first_person,
)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """
    Save mono float32 audio in [-1, 1] as 16-bit PCM WAV.
    """
    path = path.resolve()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    cfg = CONFIG
    speaker_ref_dir: Path = cfg.paths.speaker_ref_dir
    sample_rate: int = cfg.audio.sample_rate

    print(f"[TEST] Using speaker_ref_dir: {speaker_ref_dir}")
    print(f"[TEST] Using sample_rate:    {sample_rate}")

    vc_cfg = VoiceCrystalConfig(
        speaker_ref_dir=speaker_ref_dir,
        sample_rate=sample_rate,
        mode=getattr(cfg.voice, "mode", "clone_only"),
        max_latency_ms=getattr(cfg.voice, "max_latency_ms", 350.0),
    )
    voice = VoiceCrystal(vc_cfg)

    test_sentences: List[str] = [
        "I am calm and safe when I practice my words.",
        "I can say my words clearly and I feel proud of myself.",
        "I take a breath and I try again when I get stuck.",
        "I hear my own voice sounding strong and confident.",
    ]

    all_audio: List[np.ndarray] = []
    latencies_ms: List[float] = []

    for i, text in enumerate(test_sentences, start=1):
        text_fp = enforce_first_person(text)
        print(f"\n[TEST] {i}. \"{text_fp}\"")

        t0 = time.perf_counter()
        try:
            audio = voice.synthesize(text_fp, style="inner")
        except Exception as e:
            print(f"[TEST] ERROR during synthesis: {e}")
            continue
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if audio is None or len(audio) == 0:
            print("[TEST] Got empty / None audio (clone unavailable or no speaker refs).")
            continue

        audio = np.asarray(audio, dtype=np.float32)
        dur_s = float(len(audio)) / float(sample_rate)
        peak = float(np.max(np.abs(audio)))

        latencies_ms.append(dt_ms)
        all_audio.append(audio)

        print(f"[TEST] Duration: {dur_s:.2f}s, latency: {dt_ms:.1f} ms, peak: {peak:.3f}")

    if not all_audio:
        print("\n[TEST] No audio segments produced – check that:")
        print("  - XTTS is installed and loading correctly, and")
        print("  - speaker_ref_dir contains WAV files for the child voice.")
        return

    silence = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
    concat: List[np.ndarray] = []
    for seg in all_audio:
        concat.append(seg)
        concat.append(silence)
    final_audio = np.concatenate(concat, axis=0)

    out_path = Path("clone_test.wav")
    save_wav(out_path, final_audio, sample_rate)

    print("\n[TEST] Wrote:", out_path.resolve())
    if latencies_ms:
        avg_latency = sum(latencies_ms) / len(latencies_ms)
        worst_latency = max(latencies_ms)
        print(f"[TEST] Avg latency:   {avg_latency:.1f} ms")
        print(f"[TEST] Worst latency: {worst_latency:.1f} ms")


if __name__ == "__main__":
    main()
