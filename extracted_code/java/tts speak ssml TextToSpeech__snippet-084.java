  }

  private fun speakCorrectedText(text: String, prosody: Prosody) {
      if (text.isEmpty()) return
      val pitchStr = "${prosody.pitch}Hz"
      val rateStr = prosody.rate.toString()
      val volumeStr = when {
          prosody.volume > 0.6 -> "loud"
          prosody.volume < 0.4 -> "soft"
          else -> "medium"
      }
      // Add emphasis for variance/intoration
      val ssml = "<speak><prosody pitch=\"$pitchStr\" rate=\"$rateStr\"
