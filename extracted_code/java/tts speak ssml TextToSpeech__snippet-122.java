      updateVadFromPrefs()
  }

  private fun updateVadFromPrefs() {
      vadMode = sharedPrefs.getInt("vad_mode", 1)
      silenceDurationMs = sharedPrefs.getInt("silence_ms", 1200)
      vadThreshold = sharedPrefs.getFloat("vad_threshold", 0.5f)
  }

  fun tuneVAD(mode: Int, silenceMs: Int, threshold: Float) {
      vadMode = mode
      silenceDurationMs = silenceMs
      vadThreshold = threshold
  }

  fun listenAndProcess(callback: (String, Prosody) -> Unit) {
      if (ContextCompat.checkSelfPermission(context,
