  private var audioRecord: AudioRecord? = null
  private val audioBuffer = ByteArrayOutputStream()
  private val ttsMap = mutableMapOf<String, OfflineTts>()
  private val models = mutableMapOf<String, Model>()
  private val languageDetector =
