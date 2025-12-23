  }

  private fun speakCorrectedText(text: String, prosody: Prosody) {
      if (text.isEmpty() || offlineTts == null) return

      // Synthesize with default
      val generatedAudio = offlineTts!!.generate(text, speed = 1.0f,
