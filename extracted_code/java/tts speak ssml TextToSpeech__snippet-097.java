  private var audioRecord: AudioRecord? = null
  private var voskModel: Model? = null
  private val audioBuffer = ByteArrayOutputStream()
  private val ttsMap: Map<String, OfflineTts> = mapOf()
  private val languageDetector =
