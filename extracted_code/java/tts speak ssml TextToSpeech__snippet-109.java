              val correctedText = correctToFirstPerson(rawText)
              speakCorrectedText(correctedText, prosody)
              callback(correctedText, prosody)
          }
      })
      whisper.start()
  }

  private fun calculateRMS(buffer: ShortArray): Double {
      var sum = 0.0
      buffer.forEach { sum += (it / 32768.0).pow(2) }
      return if (buffer.isNotEmpty()) sqrt(sum / buffer.size) else 0.0
  }

  private fun correctToFirstPerson(text: String): String {
      if (text.isEmpty()) return ""
      var corrected = text.replace(Regex("\\b(you|he|she|they)\\b",
