  private val audioBuffer = ByteArrayOutputStream()
  private val ttsMap = mutableMapOf<String, OfflineTts>()
  private val languageDetector =
