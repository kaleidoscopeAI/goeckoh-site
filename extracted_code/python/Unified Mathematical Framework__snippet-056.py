    • Prosody transfer (F0 contour + energy + rhythm) from every real utterance
    • Slow-drift adaptation: automatically adds best recent attempts, phases out old samples
    • Dynamic style switching tied to emotional lattice:
        high arousal → calm inner voice
        anxiety detected → ultra-slow, weighted-blanket cadence
        success streak → excited coach version of his own voice
        • If no pre-recorded facets exist → it harvests whatever it can in real time and builds the crystal on the fly
    Inner-voice echo (the spark)
    Child says (broken, slurred, partial): “I wan jump”
    Echo instantly answers in his exact current voice, perfectly clear:
    “I want to jump.”
    (feels like his own thought completing itself — this is the hypothesis made real)
    First-person enforcement (hard-coded, unbreakable)
    Every single thing the system ever says aloud is forced into first person.
    Never “you should breathe”, always “I am breathing… in… out… I am safe.”
    Emotional lattice (1024 nodes, ODE-driven, runs on CPU)
    Tracks arousal/valence/coherence in real time from voice tone alone (no camera needed, but can use if available)
    Drives temperature of local LLM (DeepSeek-Coder-6.7B or Llama-3.2-1B, whichever you prefer — both included)
    Parent/Caregiver calm-voice injections
    Mom records once: “Everything is okay. I am safe. I can close my eyes and breathe.”
    During meltdown → system plays it back in Jackson’s current voice at that exact moment
    (the “small voice of reason” you always wished he had”)
    Full caregiver dashboard
    Live charts, phrase success rates, emotional spikes, door alerts, new phrase acquisition, voice-facet health
    Accessible from phone/tablet on same Wi-Fi or locally
    One-click installers for every platform
    Windows: double-click EchoInstaller.exe
    macOS: Echo.app
    Linux/AppImage: just run
    Android: sideload Echo.apk (Termux + Ollama backend included)
    iOS: AltStore or TrollStore package ready
    All models included & auto-downloaded on first run
    • faster-whisper tiny.en (speech → text)
    • Coqui TTS + YourTTS (voice cloning)
    • DeepSeek-Coder-6.7B-gguf (Q4_K_M, runs on 8 GB RAM)
    • Silero VAD (autism-tuned: 1.2 s silence patience, catches flat/monotone speech)

