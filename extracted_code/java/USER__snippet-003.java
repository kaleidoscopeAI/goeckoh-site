private val sampleRate = 16000
private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT) * 4
private val audioBuffer = ByteArrayOutputStream()
private val ttsMap = mutableMapOf<String, OfflineTts>()
private val languageDetector = LanguageDetectorBuilder.fromLanguages(Language.ENGLISH, Language.SPANISH, Language.FRENCH).build()
private lateinit var whisper: Whisper
private lateinit var recorder: Recorder
private lateinit var sileroVad: SileroVad
private lateinit var sharedPrefs: SharedPreferences
private var nsWrapper: NsWrapper? = null

// Optimized buffers
private var shortBuffer: ShortArray? = null
private var processedBuffer: ShortArray? = null
private var floatBuffer: FloatArray? = null

// VAD state
private var silenceStart: Long = 0
private var isSpeechDetected = false
private var vadThreshold: Float = 0.5f
private var silenceDurationMs: Int = 1200
private var vadMode: Int = 2
private var useNoiseSuppression: Boolean = true

// Metrics
private var totalFrames = 0
private var speechFrames = 0
private var vadLatencySum = 0L
private var vadStartTime: Long = 0
private var lastMetricsUpdate: Long = 0

private val localBroadcastManager by lazy { LocalBroadcastManager.getInstance(context) }

init {
    initializeComponents()
}

private fun initializeComponents() {
    // Initialize TTS
    initializeTTS()

    // Initialize Whisper
    initializeWhisper()

    // Initialize Recorder
    recorder = Recorder(context)

    // Initialize Silero VAD
    initializeSileroVAD()

    // Initialize Noise Suppression
    initializeNoiseSuppression()

    // Load preferences
    sharedPrefs = context.getSharedPreferences("vad_prefs", Context.MODE_PRIVATE)
    updateVadFromPrefs()

    // Pre-allocate buffers
    initializeBuffers()
}

private fun initializeTTS() {
    val ttsLangs = listOf("en", "es", "fr")
    ttsLangs.forEach { lang ->
        val modelName = when (lang) {
            "en" -> "vits-piper-en_US-amy-medium.onnx"
            "es" -> "vits-piper-es_ES-mls_9972-medium.onnx"
            "fr" -> "vits-piper-fr_FR-upmc-medium.onnx"
            else -> ""
        }
        if (modelName.isNotEmpty()) {
            val ttsModelPath = File(context.filesDir, modelName).absolutePath
            try {
                val config = OfflineTtsConfig(model = ttsModelPath, numThreads = 1, debug = false)
                ttsMap[lang] = OfflineTts(config)
            } catch (e: Exception) {
                Log.e("Mirror", "TTS initialization failed for $lang: ${e.message}")
            }
        }
    }
}

private fun initializeWhisper() {
    whisper = Whisper(context)
    val modelPath = File(context.filesDir, "whisper-tiny.tflite").absolutePath
    val vocabPath = File(context.filesDir, "filters_vocab_multilingual.bin").absolutePath
    try {
        whisper.loadModel(modelPath, vocabPath, true)
    } catch (e: Exception) {
        Log.e("Mirror", "Whisper initialization failed: ${e.message}")
    }
}

private fun initializeSileroVAD() {
    try {
        sileroVad = SileroVad.load(context)
        // Configure for optimal performance
        sileroVad.setConfig(
            sampleRate = 16000,
            frameSize = 512,
            threshold = vadThreshold,
            minSpeechDuration = 100,
            maxSpeechDuration = 10000,
            minSilenceDuration = 200
        )
    } catch (e: Exception) {
        Log.e("Mirror", "Silero VAD initialization failed: ${e.message}")
        throw e
    }
}

private fun initializeNoiseSuppression() {
    if (useNoiseSuppression) {
        try {
            System.loadLibrary("webrtc_ns_jni")
            nsWrapper = NsWrapper().apply {
                if (nativeHandle == 0L) {
                    throw RuntimeException("Failed to initialize noise suppression")
                }
            }
            Log.d("Mirror", "Noise suppression initialized successfully")
        } catch (e: UnsatisfiedLinkError) {
            Log.w("Mirror", "WebRTC NS JNI library not available, continuing without noise suppression")
            nsWrapper = null
            useNoiseSuppression = false
        } catch (e: Exception) {
            Log.e("Mirror", "Noise suppression initialization failed: ${e.message}")
            nsWrapper = null
            useNoiseSuppression = false
        }
    }
}

private fun initializeBuffers() {
    // Pre-allocate buffers for optimal performance
    val maxFrameSize = 1024
    shortBuffer = ShortArray(maxFrameSize)
    processedBuffer = ShortArray(maxFrameSize)
    floatBuffer = FloatArray(maxFrameSize)
}

private fun updateVadFromPrefs() {
    vadMode = sharedPrefs.getInt("vad_mode", 2)
    silenceDurationMs = sharedPrefs.getInt("silence_ms", 1200)
    vadThreshold = sharedPrefs.getFloat("vad_threshold", 0.5f)
    useNoiseSuppression = sharedPrefs.getBoolean("noise_suppression", true)

    // Update Silero VAD configuration
    try {
        sileroVad.setConfig(threshold = vadThreshold)
    } catch (e: Exception) {
        Log.e("Mirror", "Failed to update VAD config: ${e.message}")
    }
}

fun tuneVAD(mode: Int, silenceMs: Int, threshold: Float) {
    vadMode = mode
    silenceDurationMs = silenceMs
    vadThreshold = threshold.coerceIn(0.1f, 0.9f)

    try {
        sileroVad.setConfig(threshold = vadThreshold)
    } catch (e: Exception) {
        Log.e("Mirror", "Failed to tune VAD: ${e.message}")
    }
}

fun listenAndProcess(callback: (String, Prosody) -> Unit) {
    if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
        Log.w("Mirror", "Audio permission not granted")
        return
    }

    resetProcessingState()

    setupWhisperListener(callback)
    setupRecorderListener(callback)

    try {
        recorder.start()
        broadcastStatus("Listening...")
    } catch (e: Exception) {
        Log.e("Mirror", "Failed to start recorder: ${e.message}")
        broadcastStatus("Recorder error: ${e.message}")
    }
}

private fun resetProcessingState() {
    audioBuffer.reset()
    partialText = ""
    silenceStart = 0
    isSpeechDetected = false
    totalFrames = 0
    speechFrames = 0
    vadLatencySum = 0L
    lastMetricsUpdate = System.currentTimeMillis()
}

private fun setupWhisperListener(callback: (String, Prosody) -> Unit) {
    whisper.setAction(Whisper.ACTION_TRANSCRIBE)
    whisper.setListener(object : IWhisperListener {
        override fun onUpdateReceived(message: String) {
            partialText = message.trim()
            Log.d("Whisper", "Partial: $partialText")
        }

        override fun onResultReceived(result: String) {
            val rawText = result.trim()
            if (rawText.isNotEmpty()) {
                val prosody = extractProsody(audioBuffer.toByteArray(), rawText)
                val correctedText = correctToFirstPerson(rawText)
                speakCorrectedText(correctedText, prosody)
                callback(correctedText, prosody)
                broadcastStatus("Processed: ${correctedText.take(50)}...")
            }
        }
    })
}

private fun setupRecorderListener(callback: (String, Prosody) -> Unit) {
    recorder.setListener(object : IRecorderListener {
        override fun onUpdateReceived(message: String) {
            Log.d("Recorder", message)
        }

        override fun onDataReceived(samples: FloatArray) {
            processAudioFrame(samples)
        }
    })
}

private fun processAudioFrame(samples: FloatArray) {
    totalFrames++

    // Convert to short array for processing
    val shortArray = convertToShortArray(samples)

    // Apply noise suppression if enabled
    val processedSamples = if (useNoiseSuppression && nsWrapper != null) {
        applyNoiseSuppression(shortArray)
    } else {
        shortArray
    }

    // Forward to Whisper
    forwardToWhisper(processedSamples)

    // Process with VAD
    processWithVAD(processedSamples)

    // Collect for prosody analysis
    collectForProsody(processedSamples)
}

private fun convertToShortArray(samples: FloatArray): ShortArray {
    return ShortArray(samples.size).apply {
        for (i in indices) {
            this[i] = (samples[i] * Short.MAX_VALUE).toShort()
        }
    }
}

private fun applyNoiseSuppression(input: ShortArray): ShortArray {
    return try {
        val output = ShortArray(input.size)
        nsWrapper!!.nativeProcess(input, null, output, null)
        output
    } catch (e: Exception) {
        Log.e("Mirror", "Noise suppression failed: ${e.message}")
        input // Fallback to original audio
    }
}

private fun forwardToWhisper(samples: ShortArray) {
    // Convert back to float for Whisper
    floatBuffer?.let { floatArray ->
        val minSize = minOf(floatArray.size, samples.size)
        for (i in 0 until minSize) {
            floatArray[i] = samples[i].toFloat() / Short.MAX_VALUE
        }
        whisper.writeBuffer(floatArray.copyOf(minSize))
    }
}

private fun processWithVAD(samples: ShortArray) {
    val frameSize = 512
    var offset = 0

    while (offset + frameSize <= samples.size) {
        val frame = samples.copyOfRange(offset, offset + frameSize)

        vadStartTime = System.currentTimeMillis()
        val speechProb = try {
            sileroVad.process(frame)
        } catch (e: Exception) {
            Log.e("Mirror", "VAD processing failed: ${e.message}")
            0.0f
        }
        val latency = System.currentTimeMillis() - vadStartTime
        vadLatencySum += latency

        val isSpeech = speechProb > vadThreshold
        handleSpeechDetection(isSpeech)

        offset += frameSize
    }

    // Update metrics periodically
    if (System.currentTimeMillis() - lastMetricsUpdate > 1000) {
        logAndBroadcastVadMetrics()
        lastMetricsUpdate = System.currentTimeMillis()
    }
}

private fun handleSpeechDetection(isSpeech: Boolean) {
    if (isSpeech) {
        speechFrames++
        isSpeechDetected = true
        silenceStart = 0
    } else if (isSpeechDetected) {
        if (silenceStart == 0L) silenceStart = System.currentTimeMillis()
        if (System.currentTimeMillis() - silenceStart >= silenceDurationMs) {
            recorder.stop()
            whisper.start()
            logAndBroadcastVadMetrics()
        }
    }
}

private fun collectForProsody(samples: ShortArray) {
    val byteArray = ByteArray(samples.size * 2)
    ByteBuffer.wrap(byteArray).asShortBuffer().put(samples)
    audioBuffer.write(byteArray)
}

private fun logAndBroadcastVadMetrics() {
    val avgLatency = if (totalFrames > 0) vadLatencySum / totalFrames else 0
    val speechRate = if (totalFrames > 0) speechFrames.toFloat() / totalFrames else 0f
    val efficiency = (speechFrames.toFloat() / totalFrames) * 100

    Log.d("VAD Metrics", 
        "Frames: $totalFrames, Speech: $speechFrames, " +
        "Rate: ${"%.1f".format(speechRate * 100)}%, " +
        "Latency: ${avgLatency}ms, " +
        "Efficiency: ${"%.1f".format(efficiency)}%"
    )

    val metrics = """
        VAD Performance:
        Frames: $totalFrames/$speechFrames
        Speech Rate: ${"%.1f".format(speechRate * 100)}%
        Avg Latency: ${avgLatency}ms
        Efficiency: ${"%.1f".format(efficiency)}%
        Mode: ${when(vadMode) {
            0 -> "Ultra Patient"
            1 -> "Patient" 
            2 -> "Normal"
            3 -> "Aggressive"
            4 -> "Ultra Aggressive"
            else -> "Custom"
        }}
    """.trimIndent()

    val intent = Intent("VAD_METRICS_UPDATE").putExtra("metrics", metrics)
    localBroadcastManager.sendBroadcast(intent)
}

private fun broadcastStatus(status: String) {
    val intent = Intent("SYSTEM_STATUS_UPDATE").putExtra("status", status)
    localBroadcastManager.sendBroadcast(intent)
}

// Other functions (correctToFirstPerson, extractProsody, speakCorrectedText, playAudio) remain similar
// but with added error handling and optimization

fun cleanup() {
    try {
        nsWrapper?.nativeFree()
    } catch (e: Exception) {
        Log.e("Mirror", "Error cleaning up noise suppression: ${e.message}")
    }

    try {
        sileroVad.close()
    } catch (e: Exception) {
        Log.e("Mirror", "Error closing VAD: ${e.message}")
    }

    shortBuffer = null
    processedBuffer = null
    floatBuffer = null
}
