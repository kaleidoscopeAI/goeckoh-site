# bubble_synthesizer.py
from typing import Dict

import numpy as np

from voice_profile import VoiceFingerprint
from phoneme_utils import compute_spikiness_envelope


def feed_text_through_bubble(
    text: str,
    profile: VoiceFingerprint,
    control_rate: float = 100.0,
) -> Dict[str, np.ndarray]:
    """
    Generate deterministic control curves from text + VoiceFingerprint.

    Returns dict with:
      "t"      : time axis (s)
      "f0"     : Hz
      "energy" : 0-1
      "zcr"    : 0-1 (phoneme-aware Bouba/Kiki proxy)
      "tilt"   : spectral tilt / metalness proxy
    """
    # ----- DURATION: tied to syllables vs rate -----
    # crude syllable estimate: count vowel groups
    low = text.lower()
    syllables = 0
    in_vowel = False
    for ch in low:
        if ch in "aeiouy":
            if not in_vowel:
                syllables += 1
                in_vowel = True
        else:
            in_vowel = False
    approx_syllables = max(1, syllables)

    duration = approx_syllables / max(profile.rate, 0.5)
    control_rate = float(control_rate)
    num_steps = int(duration * control_rate)
    if num_steps < 5:
        num_steps = 5
        duration = num_steps / control_rate

    t = np.linspace(0.0, duration, num_steps)

    # ----- F0 contour (arch + jitter) -----
    phase = np.linspace(0, np.pi, num_steps)
    f0_base = profile.mu_f0 + profile.sigma_f0 * 0.5 * np.sin(phase)

    rng = np.random.RandomState(abs(hash(text)) % (2**32))
    jitter = rng.normal(0.0, profile.jitter_base * profile.sigma_f0, size=num_steps)
    f0 = f0_base + jitter

    # ----- Energy envelope (attack-sustain-decay) -----
    attack_len = max(1, int(0.1 * num_steps))
    decay_len = max(1, int(0.2 * num_steps))
    sustain_len = max(1, num_steps - attack_len - decay_len)

    attack = np.linspace(0.2, 1.0, attack_len)
    sustain = np.ones(sustain_len)
    decay = np.linspace(1.0, 0.3, decay_len)
    energy = np.concatenate([attack, sustain, decay])[:num_steps]

    # ----- Phoneme-aware Bouba/Kiki spikiness -> ZCR proxy -----
    zcr = compute_spikiness_envelope(
        text=text,
        length=num_steps,
        base_sharpness=profile.base_sharpness,
        plosive_boost=0.4,
        fricative_boost=0.25,
    )

    # ----- Tilt / metalness stays mostly profile-driven -----
    tilt = np.full(num_steps, profile.base_metalness, dtype=np.float32)

    return {
        "t": t.astype(np.float32),
        "f0": f0.astype(np.float32),
        "energy": energy.astype(np.float32),
        "zcr": zcr,
        "tilt": tilt,
    }