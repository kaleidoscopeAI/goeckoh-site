  private lateinit var whisper: Whisper
  private lateinit var recorder: Recorder
  private lateinit var sileroVad: SileroVad
  private lateinit var sharedPrefs: SharedPreferences
  private var partialText = ""
  private var silenceStart: Long = 0
  private var isSpeechDetected = false
  private var vadThreshold: Float = 0.5f
  private var silenceDurationMs: Int = 1200
  private var vadMode: Int = 1 // Silero mode: 0-3, mapping from UI

  init {
      // Init TTS
      val ttsLangs = listOf("en", "es", "fr")
      ttsLangs.forEach { lang ->
          val modelName = when (lang) {
              "en" -> "vits-piper-en_US-amy-medium.onnx"
              "es" -> "vits-piper-es_ES-mls_9972-medium.onnx"
              "fr" -> "vits-piper-fr_FR-upmc-medium.onnx"
              else -> ""
          }
          val ttsModelPath = File(context.filesDir, modelName).absolutePath
          val config = OfflineTtsConfig(model = ttsModelPath, numThreads =
