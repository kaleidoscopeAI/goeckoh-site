     70      private fun initTts() {
     72 -        val models = listOf(
     71 +        val modelsList = listOf(
     72              "en" to "vits-piper-en_US-amy-medium.onnx",
        ⋮
     75          )
     77 -        models.forEach { (lang, modelName) ->
     76 +        modelsList.forEach { (lang, modelName) ->
     77              val path = File(context.filesDir, modelName).absolutePath
        ⋮
     82
     84 -    private fun initWhisper() {
     85 -        val modelPath = File(context.filesDir, "whisper-tiny.tflite").a
         bsolutePath
     86 -        val vocabPath = File(context.filesDir, "filters_vocab_multiling
         ual.bin").absolutePath
     87 -        whisper.loadModel(modelPath, vocabPath, true)
     88 -    }
     89 -
     90 -    private fun initNoiseSuppression() {
     91 -        useNoiseSuppression = sharedPrefs.getBoolean("noise_suppression
         ", true)
     92 -        if (useNoiseSuppression) {
     93 -            try {
     94 -                System.loadLibrary("webrtc_ns_jni")
     95 -                nsWrapper = NsWrapper()
     96 -            } catch (e: UnsatisfiedLinkError) {
     97 -                Log.w("Mirror", "Noise suppression native lib missing;
         continuing without NS")
     98 -                useNoiseSuppression = false
     83 +    private fun initVosk() {
     84 +        org.kaldi.Vosk.init(context)
     85 +        val voskDirs = listOf(
     86 +            "en" to "vosk-model-small-en-us",
     87 +            "es" to "vosk-model-small-es",
     88 +            "fr" to "vosk-model-small-fr"
     89 +        )
     90 +        voskDirs.forEach { (lang, dir) ->
     91 +            val path = File(context.filesDir, dir)
     92 +            if (path.exists()) {
     93 +                models[lang] = Model(path.absolutePath)
     94 +            } else {
     95 +                Log.w("Mirror", "Vosk model missing for $lang at ${path
         .absolutePath}")
     96              }
        ⋮
    100      private fun updateVadFromPrefs() {
    104 -        vadMode = sharedPrefs.getInt("vad_mode", 2)
    101 +        vadModeIndex = sharedPrefs.getInt("vad_mode", 2)
    102          silenceDurationMs = sharedPrefs.getInt("silence_ms", 1200)
    103          vadThreshold = sharedPrefs.getFloat("vad_threshold", 0.5f)
    107 -        try {
    108 -            sileroVad.setConfig(threshold = vadThreshold)
    109 -        } catch (e: Exception) {
    110 -            Log.e("Mirror", "Unable to update VAD threshold: ${e.messag
         e}")
    104 +        rebuildVad()
    105 +    }
    106 +
    107 +    private fun rebuildVad() {
    108 +        vad.close()
    109 +        val mode = when (vadModeIndex) {
    110 +            0 -> Mode.LOW_BITRATE
    111 +            1 -> Mode.NORMAL
    112 +            3 -> Mode.AGGRESSIVE
    113 +            4 -> Mode.VERY_AGGRESSIVE
    114 +            else -> Mode.NORMAL
    115          }
    116 +        vad = VadWebRTC.builder()
    117 +            .setSampleRate(SampleRate.SAMPLE_RATE_16K)
    118 +            .setFrameSize(FrameSize.FRAME_SIZE_320)
    119 +            .setMode(mode)
    120 +            .setSilenceDurationMs(silenceDurationMs)
    121 +            .setSpeechDurationMs(100)
    122 +            .build()
    123      }
        ⋮
    125      fun tuneVAD(mode: Int, silenceMs: Int, threshold: Float) {
    115 -        vadMode = mode
    126 +        vadModeIndex = mode
    127          silenceDurationMs = silenceMs
    128          vadThreshold = threshold.coerceIn(0.1f, 0.9f)
    118 -        try {
    119 -            sileroVad.setConfig(threshold = vadThreshold)
    120 -        } catch (e: Exception) {
    121 -            Log.e("Mirror", "VAD tune failed: ${e.message}")
    122 -        }
    129 +        rebuildVad()
    130      }

