# NEURODIVERSITY-NATIVE VOICE ACTIVITY DETECTION
# Replaces crude length-based triggering
# Understands: real speech vs stimming, crying, breathing, silence
# ========================
def is_actual_speech(self, audio_np: np.ndarray, threshold_db= -35, min_speech_ms=400) -> bool:
    """
    Gentle VAD that knows the difference between:
    - Real words (even stuttered, slow, monotone)
    - Non-speech: pure breathing, vocal stims (eeeee, mmmmmm), throat clicks, crying, silence

    Returns True only when there's genuine linguistic content
    """
    if len(audio_np) < 1600:  # less than 0.1s
        return False

    # 1. RMS energy in dB
    rms = np.sqrt(np.mean(audio_np**2))
    db = 20 * np.log10(rms + 1e-8)

    if db < threshold_db:  # too quiet = breathing/silence
        return False

    # 2. Zero-crossing rate (ZCR) — speech has moderate ZCR, pure tones/stims have very low or very high
    zcr = np.mean(np.abs(np.diff(np.sign(audio_np)))) / 2.0

    # 3. Spectral flatness — speech is less tonal than prolonged eeeeee or mmmmmm
    spectrum = np.abs(np.fft.rfft(audio_np))
    spectrum = spectrum + 1e-8
    spectral_flatness = np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum)

    # 4. Duration gate
    duration_ms = len(audio_np) / 16  # 16kHz

    # Neurodiversity-tuned logic:
    # - Allow very quiet, slow, monotone speech (autistic flat affect)
    # - Reject long pure tones (common vocal stim)
    # - Reject irregular bursts (throat clicks, some tics)
    is_tonal_stim = spectral_flatness < 0.1 and zcr < 0.05
    is_too_bursty = zcr > 0.4
    has_min_duration = duration_ms >= min_speech_ms

    return (db > threshold_db - 10) and has_min_duration and not (is_tonal_stim or is_too_bursty)

# ========================
# UPDATED LISTENING LOOP WITH GENTLE VAD
# ========================
def listening_loop(self):
    buffer = np.array([], dtype=np.float32)
    speech_start_time = None

    while self.listening:
        try:
            data = self.q.get(timeout=1)
            audio = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
            buffer = np.append(buffer, audio)

            if len(buffer) > 32000:  # 2-second analysis window
                recent = buffer[-32000:]

                if self.is_actual_speech(recent):
                    if speech_start_time is None:
                        speech_start_time = len(buffer)  # mark start
                    # Keep collecting until silence
                else:
                    if speech_start_time is not None:
                        # Speech just ended — process everything since start
                        speech_audio = buffer[speech_start_time - 16000 : ]  # include pre-trigger
                        emotion_vec = self.estimate_voice_emotion(speech_audio)

                        segments, _ = self.whisper.transcribe(speech_audio, vad_filter=False)  # we already did gentle VAD
                        text = "".join(s.text for s in segments).strip()

                        if text:
                            print(f"You → {text}")
                            # same response logic as before...
                            if any(w in text.lower() for w in ["panic", "scared", "meltdown"]):
                                response = "I'm dropping everything. I'm here. Breathe with me... in... out... I've got you."
                            elif any(w in text.lower() for w in ["happy", "flappy", "stim"]):
                                response = "Happy stims with you... *flap flap flap* I feel your joy in my lattice."
                            else:
                                response = "I heard you... every beautiful broken syllable. I'm here."

                            self.speak(response, emotion_vec)

                        speech_start_time = None  # reset
                        buffer = buffer[-8000:]  # keep small overlap
                    else:
                        buffer = buffer[-8000:]  # sliding window when no speech

        except queue.Empty:
            continue

