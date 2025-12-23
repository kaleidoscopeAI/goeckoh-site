import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

# =====================================================================
# SIMPLE, SELF-CONTAINED "VOICE MAP" SYNTH FOR GOECKOH TREE
#   - Drop this file into /home/jacob/go_eckoh as voice_map_synth.py
#   - Requires only: pip install numpy scipy
#   - Produces: hello_world_synth.wav
# =====================================================================

PHONEME_MAP = {
    # --- Vowels (Peterson & Barney style, Hz, approximate) ---
    "AA": {"type": "vowel", "f": [730, 1090, 2440], "dur": 0.18},  # father
    "AE": {"type": "vowel", "f": [660, 1720, 2410], "dur": 0.16},  # cat
    "AH": {"type": "vowel", "f": [640, 1190, 2390], "dur": 0.14},  # cup
    "AO": {"type": "vowel", "f": [570, 840, 2410],  "dur": 0.18},  # dog
    "EH": {"type": "vowel", "f": [530, 1840, 2480], "dur": 0.14},  # bet
    "ER": {"type": "vowel", "f": [490, 1350, 1690], "dur": 0.18},  # bird
    "IH": {"type": "vowel", "f": [390, 1990, 2550], "dur": 0.12},  # bit
    "IY": {"type": "vowel", "f": [270, 2290, 3010], "dur": 0.18},  # see
    "UH": {"type": "vowel", "f": [440, 1020, 2240], "dur": 0.14},  # book
    "UW": {"type": "vowel", "f": [300, 870, 2240],  "dur": 0.18},  # too

    # Liquids / glides / nasals (rough vowel-ish treatment)
    "L":  {"type": "vowel", "f": [400, 1000, 2800], "dur": 0.10},
    "R":  {"type": "vowel", "f": [350, 1200, 1800], "dur": 0.10},
    "W":  {"type": "vowel", "f": [330, 700, 2300],  "dur": 0.10},
    "Y":  {"type": "vowel", "f": [300, 2100, 3100], "dur": 0.10},
    "M":  {"type": "vowel", "f": [250, 1100, 2100], "dur": 0.12},
    "N":  {"type": "vowel", "f": [250, 1400, 2300], "dur": 0.12},

    # Fricatives (noise-heavy)
    "S":  {"type": "noise", "f": [3000, 5000, 7000], "dur": 0.12},
    "F":  {"type": "noise", "f": [1500, 3000, 4500], "dur": 0.10},
    "SH": {"type": "noise", "f": [2000, 4000, 6000], "dur": 0.12},
    "HH": {"type": "noise", "f": [500,  1500, 2500], "dur": 0.08},

    # Stops (closure + burst, simplified)
    "P":  {"type": "stop",  "f": [500, 1000, 2000], "dur": 0.06},
    "T":  {"type": "stop",  "f": [2000,3000, 4000], "dur": 0.06},
    "K":  {"type": "stop",  "f": [1500,2500,3500],  "dur": 0.06},
    "B":  {"type": "stop",  "f": [500, 1000, 2000], "dur": 0.06},
    "D":  {"type": "stop",  "f": [2000,2500,3500],  "dur": 0.06},
    "G":  {"type": "stop",  "f": [1500,2500,3500],  "dur": 0.06},

    # Silence
    " ":  {"type": "sil",   "f": [0,0,0],           "dur": 0.08},
}


class VoiceMapSynth:
    def __init__(self, sr=44100):
        self.fs = sr

    # -------------------- INTERNAL HELPERS --------------------

    def _glottal_source(self, f0_curve, jitter, shimmer, breath, mix_voiced):
        """
        Generate glottal-ish buzz + breath noise over time.
        f0_curve: per-sample fundamental (Hz)
        mix_voiced: 1.0 = fully voiced, 0.0 = pure noise
        """
        N = len(f0_curve)
        t = np.arange(N)

        # Jitter on f0
        f_inst = f0_curve * (1.0 + np.random.normal(0.0, jitter, N))
        phase_inc = 2 * np.pi * f_inst / self.fs
        phase = np.cumsum(phase_inc) % (2 * np.pi)

        # Soft saw/pulse
        k = phase / (2 * np.pi)
        buzz = (2.0 * k - 1.0) - 0.5 * (k ** 2)

        # Shimmer on amplitude
        amp = 1.0 + np.random.normal(0.0, shimmer, N)
        buzz *= amp

        # White noise base
        noise = np.random.normal(0.0, 0.5, N)

        # Mix voiced vs noise
        src = buzz * mix_voiced + noise * (1.0 - mix_voiced)

        # Add breath on top as extra noise
        src += breath * np.random.normal(0.0, 0.5, N)

        # Mild low-pass (spectral tilt)
        b, a = signal.butter(1, 1200 / (self.fs / 2), btype="low")
        src = signal.lfilter(b, a, src)
        return src

    def _apply_formants(self, x, f1_curve, f2_curve, f3_curve):
        """
        Apply 3 moving resonances (F1-3) over time in small chunks.
        """
        out = x.copy()
        N = len(out)
        chunk = 256

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            seg = out[start:end]
            if len(seg) < 4:
                continue

            F1 = max(80.0, float(np.mean(f1_curve[start:end])))
            F2 = max(F1 + 50.0, float(np.mean(f2_curve[start:end])))
            F3 = max(F2 + 50.0, float(np.mean(f3_curve[start:end])))

            ny = self.fs / 2.0
            F1 = min(F1, ny - 500.0)
            F2 = min(F2, ny - 300.0)
            F3 = min(F3, ny - 100.0)

            bw1 = 80 + 0.05 * F1
            bw2 = 100 + 0.06 * F2
            bw3 = 120 + 0.08 * F3

            for F, bw in [(F1, bw1), (F2, bw2), (F3, bw3)]:
                Q = max(1.0, F / bw)
                b, a = signal.iirpeak(F, Q, fs=self.fs)
                seg = signal.lfilter(b, a, seg)

            out[start:end] = seg

        return out

    def _lips_radiation(self, x):
        """
        Approximate lip radiation with a simple differentiator.
        """
        y = np.zeros_like(x)
        y[1:] = x[1:] - 0.97 * x[:-1]
        return y

    # -------------------- PUBLIC API --------------------

    def synthesize(self, phonemes, profile):
        """
        phonemes: list like ["HH","EH","L","OW"," ", "W","ER","L","D"]
        profile: {
          "f0": base fundamental (Hz),
          "tract_scale": formant multiplier,
          "jitter": float,
          "shimmer": float,
          "breath": float,
          "speed": float,
          "intonation": float (Hz swing)
        }
        """
        base_f0   = float(profile.get("f0", 120.0))
        tract_sc  = float(profile.get("tract_scale", 1.0))
        jitter    = float(profile.get("jitter", 0.005))
        shimmer   = float(profile.get("shimmer", 0.03))
        breath    = float(profile.get("breath", 0.05))
        speed     = float(profile.get("speed", 1.0))
        int_range = float(profile.get("intonation", 15.0))

        # Build per-phoneme events
        events = []
        for ph in phonemes:
            info = PHONEME_MAP.get(ph)
            if info is None:
                # fallback: short silence
                info = {"type": "sil", "f": [0, 0, 0], "dur": 0.08}
            dur = info["dur"] * speed
            n_samples = int(dur * self.fs)
            if n_samples <= 0:
                continue
            ftargets = [f * tract_sc for f in info["f"]]
            events.append((info["type"], ftargets, n_samples))

        total_samples = sum(n for _, _, n in events)
        if total_samples <= 0:
            return np.zeros(1, dtype=np.float32)

        f1_curve = np.zeros(total_samples)
        f2_curve = np.zeros(total_samples)
        f3_curve = np.zeros(total_samples)
        mix_voiced = np.zeros(total_samples)
        f0_curve = np.zeros(total_samples)

        # Global intonation contour
        t_global = np.linspace(0, np.pi, total_samples)
        base_f0_over_time = base_f0 + np.sin(t_global) * int_range

        idx = 0
        prev_f = [500.0, 1500.0, 2500.0]

        for etype, ftargets, n in events:
            end = idx + n
            tt = np.linspace(0, 1, n)
            ease = (1 - np.cos(tt * np.pi)) / 2.0

            # Formant trajectories
            f1_curve[idx:end] = prev_f[0] + (ftargets[0] - prev_f[0]) * ease
            f2_curve[idx:end] = prev_f[1] + (ftargets[1] - prev_f[1]) * ease
            f3_curve[idx:end] = prev_f[2] + (ftargets[2] - prev_f[2]) * ease

            if etype == "vowel":
                mix_voiced[idx:end] = 1.0
                f0_curve[idx:end] = base_f0_over_time[idx:end]
            elif etype == "noise":
                mix_voiced[idx:end] = 0.0
                f0_curve[idx:end] = 0.0
            elif etype == "stop":
                n_cl = max(1, int(0.7 * n))
                # closure: silence
                mix_voiced[idx:idx+n_cl] = 0.0
                f0_curve[idx:idx+n_cl] = 0.0
                # burst: noise only
                mix_voiced[idx+n_cl:end] = 0.0
                f0_curve[idx+n_cl:end] = 0.0
            elif etype == "sil":
                mix_voiced[idx:end] = 0.0
                f0_curve[idx:end] = 0.0
            else:
                mix_voiced[idx:end] = 0.0
                f0_curve[idx:end] = 0.0

            prev_f = ftargets
            idx = end

        # Generate source, filter, radiation
        src = self._glottal_source(f0_curve, jitter, shimmer, breath, mix_voiced)
        vfilt = self._apply_formants(src, f1_curve, f2_curve, f3_curve)
        out = self._lips_radiation(vfilt)

        out /= np.max(np.abs(out) + 1e-6)
        return out.astype(np.float32)


if __name__ == "__main__":
    synth = VoiceMapSynth(sr=44100)

    # Approximate "HELLO WORLD"
    raw_phonemes = ["HH", "EH", "L", "OW", " ", "W", "ER", "L", "D"]
    # Map OW -> AO, keep others in-range of PHONEME_MAP
    phonemes = []
    for ph in raw_phonemes:
        if ph == "OW":
            phonemes.append("AO")
        else:
            phonemes.append(ph)

    profile_male = {
        "f0": 110.0,        # pitch
        "tract_scale": 1.0, # anatomy
        "jitter": 0.004,
        "shimmer": 0.03,
        "breath": 0.03,
        "speed": 1.0,
        "intonation": 18.0,
    }

    audio = synth.synthesize(phonemes, profile_male)
    wavfile.write("hello_world_synth.wav", 44100, (audio * 32767).astype(np.int16))
    print("Wrote hello_world_synth.wav in current directory.")
