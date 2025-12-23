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
private var vadMode: Int = 1
private var nsWrapper: NsWrapper? = null
private val localBroadcastManager = LocalBroadcastManager.getInstance(context)
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

    // Init RNNoise via JNI
    System.loadLibrary("webrtc_ns_jni")
    nsWrapper = NsWrapper()
    nsWrapper!!.nativeCreate()

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

            // Optimize buffer: Use direct ByteBuffer
            val shortBuffer = ByteBuffer.allocateDirect(samples.size * 2).asShortBuffer()
            for (i in samples.indices) {
                shortBuffer.put((samples[i] * Short.MAX_VALUE).toShort())
            }

            // RNNoise suppression
            val inShort = ShortArray(samples.size)
            shortBuffer.rewind()
            shortBuffer.get(inShort)
            val outShort = ShortArray(samples.size)
            nsWrapper?.nativeProcess(inShort, null, outShort, null)

            // Convert back to float for Whisper if needed, but assume Whisper accepts short[]
            val cleanedSamples = FloatArray(samples.size)
            for (i in outShort.indices) {
                cleanedSamples[i] = outShort[i].toFloat() / Short.MAX_VALUE
            }

            // Forward to Whisper
            whisper.writeBuffer(cleanedSamples)

            // Silero VAD: Process in 512 sample frames
            val frameSize = 512
            var offset = 0
            while (offset + frameSize <= outShort.size) {
                val frame = outShort.copyOfRange(offset, offset + frameSize)

                vadStartTime = System.currentTimeMillis()
                val speechProb = sileroVad.process(frame)
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
                        logAndBroadcastVadMetrics()
                        break
                    }
                }

                offset += frameSize
            }

            // Collect for prosody: Append cleaned byte buffer
            val byteArray = ByteArray(outShort.size * 2)
            ByteBuffer.wrap(byteArray).asShortBuffer().put(outShort)
            audioBuffer.write(byteArray)
        }
    })

    recorder.start()
}

private fun logAndBroadcastVadMetrics() {
    val avgLatency = if (totalFrames > 0) vadLatencySum / totalFrames else 0
    val speechRate = if (totalFrames > 0) speechFrames.toFloat() / totalFrames else 0f
    Log.d("VAD Metrics", "Total frames: $totalFrames, Speech frames: $speechFrames, Speech rate: $speechRate, Avg latency: $avgLatency ms")

    val intent = Intent("VAD_METRICS_UPDATE")
    intent.putExtra("metrics", "Frames: $totalFrames/$speechFrames\nRate: ${String.format("%.2f", speechRate)}\nAvg Latency: $avgLatency ms")
    localBroadcastManager.sendBroadcast(intent)
}

// Other functions unchanged
// ...

override fun finalize() {
    nsWrapper?.nativeFree()
    sileroVad.close()
}
