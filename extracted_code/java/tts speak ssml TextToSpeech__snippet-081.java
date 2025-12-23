  private var audioRecord: AudioRecord? = null
  private var voskModel: Model? = null
  private val audioBuffer = ByteArrayOutputStream()

  init {
      Vosk.init(context)
      val modelPath = File(context.filesDir, "vosk-model-en-us-0.22")
      if (!modelPath.exists()) {
          // Assume model extracted; in production, add asset copy logic
          Toast.makeText(context, "Vosk model missing",
