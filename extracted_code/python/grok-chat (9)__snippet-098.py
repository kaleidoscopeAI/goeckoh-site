    speak the backend reply text; it is not yet playing the cloned waveform from the backend.

  What to flip to actually get realtime corrected self‑voice playback

  1. Run with audio + TTS enabled (and a sound device):
      - Drop the disables: remove GOECKOH_DISABLE_TTS=1 and GOECKOH_DISABLE_RTVC=1.
      - Keep GOECKOH_ASR_BACKEND=sherpa (or vosk) and optionally GOECKOH_PERF_MODE=1 to stay snappy.
  2. Provide / verify a clone reference:
      - Either set GOECKOH_CLONE_WAV=/path/to/your.wav or let auto-capture take the first spoken utterance.
  3. Keep audio output on:
      - Don’t set GOECKOH_DISABLE_AUDIO_PLAYBACK; ensure a working output device (or loopback on the test box).

  If you want, I can restart the backend with TTS/RTVC/audio enabled and run a live loop (ASR → correction → cloned playback) to
  confirm end‑to‑end latency and behavior.

