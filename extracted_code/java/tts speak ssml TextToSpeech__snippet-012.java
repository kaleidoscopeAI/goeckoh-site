class NeuroAcousticMirror(private val context: Context) {
    private val vadSilenceThresholdMs = 1200L
    private val sampleRate = 16000
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT) * 4
    private var audioRecord: AudioRecord? = null
    private var voskModel: Model? = null
    private val audioBuffer = ByteArrayOutputStream()
    private var offlineTts: OfflineTts? = null

    init {
        Vosk.init(context)
        val modelPath = File(context.filesDir, "vosk-model-en-us-0.22")
        if (!modelPath.exists()) {
            Toast.makeText(context, "Vosk model missing", Toast.LENGTH_SHORT).show()
        } else {
            voskModel = Model(modelPath.absolutePath)
        }

        // Init Sherpa TTS
        val ttsModelPath = File(context.filesDir, "vits-piper-en_US-amy-medium.onnx").absolutePath
        val config = TtsConfig(model = ttsModelPath, numThreads = 1, debug = false)
        offlineTts = OfflineTts(config)
    }

    fun listenAndProcess(callback: (String, Prosody) -> Unit) {
        if (voskModel == null || ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return

        audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize)
        audioRecord?.startRecording()
        audioBuffer.reset()

        val recognizer = Vosk.createRecognizer(voskModel, sampleRate.toFloat())
        recognizer.setRecognitionListener(object : RecognitionListener {
            override fun onPartialResult(hypothesis: String?) {}
            override fun onResult(hypothesis: String?) {
                val rawText = hypothesis?.let { parseJsonResult(it) } ?: return
                val prosody = extractProsody(audioBuffer.toByteArray(), rawText)
                val correctedText = correctToFirstPerson(rawText)
                speakCorrectedText(correctedText, prosody)
                callback(correctedText, prosody)
            }
            override fun onFinalResult(hypothesis: String?) {}
            override fun onError(e: Exception?) { Log.e("Mirror", "Recognition error: ${e?.message}") }
            override fun onTimeout() {}
        })

        val buffer = ByteArray(bufferSize)
        var silenceStart = 0L
        val startTime = System.currentTimeMillis()
        while (true) {
            val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
            if (read > 0) {
                audioBuffer.write(buffer, 0, read)
                val shortBuffer = ShortArray(read / 2)
                java.nio.ByteBuffer.wrap(buffer, 0, read).asShortBuffer().get(shortBuffer)
                recognizer.acceptWaveForm(shortBuffer, shortBuffer.size)
                val energy = calculateRMS(shortBuffer)
                if (energy < 0.01) {
                    if (silenceStart = 0L) silenceStart = System.currentTimeMillis()
                    if (System.currentTimeMillis() - silenceStart > vadSilenceThresholdMs && System.currentTimeMillis() - startTime > 1000) {
                        recognizer.result
                        break
                    }
                } else {
                    silenceStart = 0L
                }
            } else if (read < 0) {
                break
            }
        }

        audioRecord?.stop()
        audioRecord?.release()
    }

    private fun parseJsonResult(json: String): String {
        val textStart = json.indexOf("\"text\" : \"") + 10
        val textEnd = json.lastIndexOf("\"")
        return if (textStart > 9 && textEnd > textStart) json.substring(textStart, textEnd).trim() else ""
    }

    private fun calculateRMS(buffer: ShortArray): Double {
        var sum = 0.0
        buffer.forEach { sum += (it / 32768.0).pow(2) }
        return if (buffer.isNotEmpty()) sqrt(sum / buffer.size) else 0.0
    }

    private fun correctToFirstPerson(text: String): String {
        if (text.isEmpty()) return ""
        var corrected = text.replace(Regex("\\b(you|he|she|they)\\b", RegexOption.IGNORE_CASE), "I")
            .replace(Regex("\\b(your|his|her|their)\\b", RegexOption.IGNORE_CASE), "my")
        if (!corrected.matches(Regex("I .*", RegexOption.IGNORE_CASE))) {
            corrected = "I " + corrected.replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.getDefault()) else it.toString() }
        }
        return corrected
    }

    private fun extractProsody(audioBytes: ByteArray, text: String): Prosody {
        if (audioBytes.isEmpty()) return Prosody(120.0, 1.0, 0.5, 0.5, 20.0)

        val dispatcher = AudioDispatcherFactory.fromByteArray(audioBytes, sampleRate, 1024, 512)
        val pitches = mutableListOf<Float>()
        dispatcher.addAudioProcessor(PitchProcessor(PitchProcessor.PitchEstimationAlgorithm.YIN, sampleRate.toFloat(), 1024, object : PitchDetectionHandler {
            override fun handlePitch(result: PitchDetectionResult, event: AudioEvent) {
                if (result.pitch > 0) pitches.add(result.pitch)
            }
        }))
        dispatcher.run()
        val avgPitch = if (pitches.isNotEmpty()) pitches.average() else 120.0
        val pitchVariance = if (pitches.size > 1) pitches.map { (it - avgPitch).pow(2) }.average() else 0.0

        val durationSec = audioBytes.size / (sampleRate * 2).toDouble()
        val wordCount = text.split("\\s+".toRegex()).size.toDouble()
        val rate = if (durationSec > 0) (wordCount / durationSec) / 1.5 else 1.0

        val shortArray = ShortArray(audioBytes.size / 2)
        java.nio.ByteBuffer.wrap(audioBytes).asShortBuffer().get(shortArray)
        val volume = calculateRMS(shortArray)

        val arousal = ((avgPitch / 150.0) + (pitchVariance / 50.0) + (volume * 5.0)) / 3.0.coerceIn(0.0, 1.0)

        return Prosody(avgPitch, rate.coerceIn(0.5, 2.0), arousal, volume.coerceIn(0.0, 1.0), pitchVariance)
    }

    private fun speakCorrectedText(text: String, prosody: Prosody) {
        if (text.isEmpty() || offlineTts == null) return

        // Synthesize with default
        val generatedAudio = offlineTts!!.generate(text, speed = 1.0f, speakerId = 0) // Assume model supports speed and sid

        var samples = generatedAudio.samples // Assume float[] 
        val ttsSampleRate = generatedAudio.sampleRate // Assume 16000 or 22050

        // Post-process for prosody transfer
        val dispatcher = AudioDispatcherFactory.fromFloatArray(samples, ttsSampleRate, 1024, 512)

        // Adjust rate (time stretch)
        val rateTransposer = RateTransposer(prosody.rate)
        dispatcher.addAudioProcessor(rateTransposer)

        // Adjust pitch
        val pitchShiftFactor = prosody.pitch / 120.0 // Assume base pitch 120Hz
        val pitchShift = PitchShift(pitchShiftFactor, ttsSampleRate.toFloat())
        dispatcher.addAudioProcessor(pitchShift)

        // Adjust volume
        val gain = prosody.volume * 2.0 // Normalize to gain factor
        val gainProcessor = GainProcessor(gain)
        dispatcher.addAudioProcessor(gainProcessor)

        val processedBuffer = FloatArray(samples.size * 2) // Extra space
        var index = 0
        dispatcher.addAudioProcessor(object : AudioProcessor {
            override fun process(audioEvent: AudioEvent?): Boolean {
                val buffer = audioEvent!!.floatBuffer
                buffer.copyInto(processedBuffer, index)
                index += buffer.size
                return true
            }
            override fun processingFinished() {}
        })
        dispatcher.run()

        val processedSamples = processedBuffer.copyOf(index)

        // Play the processed audio
        playAudio(processedSamples, ttsSampleRate)
    }

    private fun playAudio(samples: FloatArray, sampleRate: Int) {
        val bufferSize = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        val audioTrack = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setSampleRate(sampleRate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setEncoding(AudioFormat.ENCODING_PCM_FLOAT).build())
            .setBufferSizeInBytes(bufferSize)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        audioTrack.play()
        audioTrack.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
        audioTrack.stop()
        audioTrack.release()
    }
