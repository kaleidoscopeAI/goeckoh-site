# voice_logger.py
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import SAMPLE_RATE, PROFILES_DIR, DEFAULT_VOICE_LABEL
from dsp_features import extract_frame_features
from voice_profile import VoiceFingerprint, get_profile_path
from speaker_embed import SpeakerEmbedder
from cloning_backend import upload_reference_audio


def record_audio(duration_sec: float) -> np.ndarray:
    print(f"[VoiceLogger] Recording {duration_sec:.1f}s of audio...")
    audio = sd.rec(
        int(duration_sec * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio[:, 0]


def _aggregate_features(y: np.ndarray, sr: int) -> VoiceFingerprint:
    feats = extract_frame_features(y, sr)
    mu_f0 = float(np.median(feats.f0))
    sigma_f0 = float(np.std(feats.f0))

    base_roughness = float(np.clip(-np.mean(feats.hnr) / 20.0 + 0.5, 0.0, 1.0))
    base_metalness = float(np.clip(np.mean(feats.tilt) / 5.0 + 0.5, 0.0, 1.0))
    base_sharpness = float(np.clip(np.mean(feats.zcr) * 5.0, 0.0, 1.0))

    energy = feats.energy
    thresh = np.mean(energy) + 0.5 * np.std(energy)
    peaks = np.where(energy > thresh)[0]
    if len(peaks) > 1:
        duration_sec = len(energy) * 0.01  # 10 ms hops
        rate = float(len(peaks) / duration_sec)
    else:
        rate = 3.0

    return VoiceFingerprint(
        mu_f0=mu_f0,
        sigma_f0=sigma_f0,
        base_roughness=base_roughness,
        base_metalness=base_metalness,
        base_sharpness=base_sharpness,
        rate=rate,
        jitter_base=0.08,
        shimmer_base=0.08,
        base_radius=1.0,
    )


def enroll_child(profile_name: str, duration_sec: float = 10.0) -> None:
    PROFILES_DIR.mkdir(exist_ok=True, parents=True)

    y = record_audio(duration_sec)
    fingerprint = _aggregate_features(y, SAMPLE_RATE)

    profile_path = get_profile_path(profile_name)
    fingerprint.to_json(profile_path)
    print(f"[VoiceLogger] Saved VoiceFingerprint to {profile_path}")

    # reference WAV for cloning
    ref_dir = PROFILES_DIR / "refs"
    ref_dir.mkdir(exist_ok=True, parents=True)
    ref_wav_path = ref_dir / f"{profile_name}_ref.wav"
    sf.write(str(ref_wav_path), y, SAMPLE_RATE)
    print(f"[VoiceLogger] Saved reference audio to {ref_wav_path}")

    # upload to OpenVoice_server
    upload_reference_audio(ref_wav_path, voice_label=profile_name or DEFAULT_VOICE_LABEL)

    # speaker embedding
    emb = SpeakerEmbedder().embed_wav(ref_wav_path)
    emb_path = PROFILES_DIR / f"{profile_name}_embed.npy"
    np.save(emb_path, emb)
    print(f"[VoiceLogger] Saved speaker embedding to {emb_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enroll a child profile for Goeckoh.")
    parser.add_argument("--name", required=True, help="Profile name (e.g. jackson)")
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    enroll_child(args.name, args.duration)
