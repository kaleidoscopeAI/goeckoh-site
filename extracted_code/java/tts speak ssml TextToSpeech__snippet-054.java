private val sampleRate = 16000
private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT) * 4
private val audioBuffer = ByteArrayOutputStream()
private val ttsMap = mutableMapOf<String, OfflineTts>()
private val languageDetector = LanguageDetectorBuilder.fromLanguages(Language.ENGLISH, Language.SPANISH, Language.FRENCH).build()
private lateinit var whisper: Whisper
private lateinit var recorder: Recorder
private lateinit var sileroVad: SileroVad
private lateinit var sharedPrefs: SharedPreferences
private var partialText = ""
private var silenceStart: Long = 0
private var isSpeechDetected = false
private var vadThreshold: Float = 0.5f
private var silenceDurationMs: Int = 1200
private var vadMode: Int = 1 // Not used in Silero, but for adaptive
private var noiseSuppressor: NoiseSuppressor? = null
// Metrics
private var totalFrames = 0
private var speechFrames = 0
private var vadLatencySum = 0L
private var vadStartTime: Long = 0

init {
    // Init TTS (unchanged)
    // ...

    // Init Whisper (unchanged)
    // ...

    // Init Recorder (unchanged)
    // ...

    // Init Silero VAD
    sileroVad = SileroVad.load(context)

    // Init Noise Suppressor
    if (NoiseSuppressor.isAvailable()) {
        noiseSuppressor = NoiseSuppressor.create(recorder.audioSessionId) // Assume recorder has audioSessionId
        noiseSuppressor?.enabled = true
    } else {
        Log.w("Mirror", "Noise suppression not available")
    }

    // Load prefs
    sharedPrefs = context.getSharedPreferences("vad_prefs", Context.MODE_PRIVATE)
    updateVadFromPrefs()
}

private fun updateVadFromPrefs() {
    vadMode = sharedPrefs.getInt("vad_mode", 1)
    silenceDurationMs = sharedPrefs.getInt("silence_ms", 1200)
    vadThreshold = sharedPrefs.getFloat("vad_threshold", 0.5f)
}

fun tuneVAD(mode: Int, silenceMs: Int, threshold: Float) {
    vadMode = mode
    silenceDurationMs = silenceMs
    vadThreshold = threshold
}

fun listenAndProcess(callback: (String, Prosody) -> Unit) {
    if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return

    audioBuffer.reset()
    partialText = ""
    silenceStart = 0
    isSpeechDetected = false
    totalFrames = 0
    speechFrames = 0
    vadLatencySum = 0L

    // Set Whisper listener (unchanged)
    // ...

    // Set Recorder listener
    recorder.setListener(object : IRecorderListener {
        override fun onUpdateReceived(message: String) {
            Log.d("Recorder", message)
        }

        override fun onDataReceived(samples: FloatArray) {
            totalFrames++

            // Apply noise suppression if available (NoiseSuppressor acts on the audio session, so assume recorder output is suppressed)
            // If not, manual suppression can be added using RNNoise lib, but skip for now

            // Optimize Silero: Process in 512 sample frames (32ms at 16kHz)
            val frameSize = 512
            var offset = 0
            while (offset + frameSize <= samples.size) {
                val frame = samples.copyOfRange(offset, offset + frameSize)
                val shortFrame = ShortArray(frameSize)
                for (i in 0 until frameSize) {
                    shortFrame[i] = (frame[i] * Short.MAX_VALUE).toShort()
                }

                vadStartTime = System.currentTimeMillis()
                val speechProb = sileroVad.process(shortFrame)
                val latency = System.currentTimeMillis() - vadStartTime
                vadLatencySum += latency

                val isSpeech = speechProb > vadThreshold
                if (isSpeech) {
                    speechFrames++
                    isSpeechDetected = true
                    silenceStart = 0
                } else if (isSpeechDetected) {
                    if (silenceStart == 0L) silenceStart = System.currentTimeMillis()
                    if (System.currentTimeMillis() - silenceStart >= silenceDurationMs) {
                        recorder.stop()
                        whisper.start()
                        logVadMetrics()
                        break
                    }
                }

                offset += frameSize
            }

            // Forward full chunk to Whisper
            whisper.writeBuffer(samples)

            // Collect for prosody
            val byteArray = ByteArray(samples.size * 2)
            ByteBuffer.wrap(byteArray).asShortBuffer().put(shortFrame) // Use last frame or full? Full
            val fullShort = ShortArray(samples.size)
            for (i in samples.indices) {
                fullShort[i] = (samples[i] * Short.MAX_VALUE).toShort()
            }
            ByteBuffer.wrap(byteArray).asShortBuffer().put(fullShort)
            audioBuffer.write(byteArray)
        }
    })

    recorder.start()
}

private fun logVadMetrics() {
    val avgLatency = if (totalFrames > 0) vadLatencySum / totalFrames else 0
    val speechRate = if (totalFrames > 0) speechFrames.toFloat() / totalFrames else 0f
    Log.d("VAD Metrics", "Total frames: $totalFrames, Speech frames: $speechFrames, Speech rate: $speechRate, Avg latency: $avgLatency ms")
    // Can add false positive estimation if ground truth available, but for now log basics
}

// Other functions unchanged
// ...

override fun finalize() {
    noiseSuppressor?.release()
    sileroVad.close()
}
