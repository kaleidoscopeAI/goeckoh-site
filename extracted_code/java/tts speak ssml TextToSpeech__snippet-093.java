  private var audioRecord: AudioRecord? = null
  private var voskModel: Model? = null
  private val audioBuffer = ByteArrayOutputStream()
  private var offlineTts: OfflineTts? = null

  init {
      Vosk.init(context)
      val modelPath = File(context.filesDir, "vosk-model-en-us-0.22")
      if (!modelPath.exists()) {
          Toast.makeText(context, "Vosk model missing",
