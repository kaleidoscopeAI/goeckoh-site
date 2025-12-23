  private lateinit var whisper: Whisper
  private lateinit var recorder: Recorder

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
