  }

  private fun calculateRMS(buffer: ShortArray): Double {
      var sum = 0.0
      buffer.forEach { sum += (it / 32768.0).pow(2) }
      return sqrt(sum / buffer.size)
  }

  private fun correctToFirstPerson(text: String): String {
      var corrected = text.replace(Regex("\\byou\\b",
