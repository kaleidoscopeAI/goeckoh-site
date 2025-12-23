  }

  private fun speakCorrectedText(text: String, prosody: Prosody) {
      if (text.isEmpty()) return

      // Detect language
      val detectedLanguage = languageDetector.detectLanguageOf(text) ?:
