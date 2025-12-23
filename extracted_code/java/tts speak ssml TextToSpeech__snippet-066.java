  private var audioRecord: AudioRecord? = null
  private var voskModel: Model? = null
  private val audioBuffer = ByteArrayOutputStream()

  init {
      val modelPath = File(context.filesDir, "vosk-model-en-us-0.22")
      voskModel = Model(modelPath.absolutePath)
  }

  fun listenAndProcess(callback: (String, Prosody) -> Unit) {
      if (ContextCompat.checkSelfPermission(context,
