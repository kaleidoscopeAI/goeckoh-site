class NeuroAcousticMirror(private val context: Context) {
    private val vadSilenceThresholdMs = 1200L
    private val sampleRate = 16000
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT) * 4
    private val audioBuffer = ByteArrayOutputStream()
    private val ttsMap = mutableMapOf<String, OfflineTts>()
    private val languageDetector = LanguageDetectorBuilder.fromLanguages(Language.ENGLISH, Language.SPANISH, Language.FRENCH).build()
    private lateinit var whisper: Whisper
    private lateinit var recorder: Recorder
    private var silenceStart: Long = 0
    private var lastEnergy: Double = 0.0
    private var partialText = ""

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
            val config = OfflineTtsConfig(model = ttsModelPath, numThreads = 1, debug = false)
            ttsMap[lang] = OfflineTts(config)
        }

        // Init Whisper
        whisper = Whisper(context)
        val modelPath = File(context.filesDir, "whisper-tiny.tflite").absolutePath
        val vocabPath = File(context.filesDir, "filters_vocab_multilingual.bin").absolutePath
        whisper.loadModel(modelPath, vocabPath, true) // multilingual = true

        // Init Recorder
        recorder = Recorder(context)
    }

    fun listenAndProcess(callback: (String, Prosody) -> Unit) {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return

        audioBuffer.reset()
        partialText = ""
        silenceStart = 0

        // Set Whisper for live transcription
        whisper.setAction(Whisper.ACTION_TRANSCRIBE) // Or specific live action if available
        whisper.setListener(object : IWhisperListener {
            override fun onUpdateReceived(message: String) {
                partialText = message.trim()
                Log.d("Whisper", "Partial: $partialText")
                // Could callback partial if needed, but for echo, wait for final
            }

            override fun onResultReceived(result: String) {
                val rawText = result.trim()
                val prosody = extractProsody(audioBuffer.toByteArray(), rawText)
                val correctedText = correctToFirstPerson(rawText)
                speakCorrectedText(correctedText, prosody)
                callback(correctedText, prosody)
            }
        })

        recorder.setListener(object : IRecorderListener {
            override fun onUpdateReceived(message: String) {
                Log.d("Recorder", message)
            }

            override fun onDataReceived(samples: FloatArray) {
                // Forward to Whisper for real-time processing
                whisper.writeBuffer(samples)

                // Collect for prosody (convert float to byte)
                val byteBuffer = ByteBuffer.allocate(samples.size * 4)
                val floatBuffer = byteBuffer.asFloatBuffer()
                floatBuffer.put(samples)
                audioBuffer.write(byteBuffer.array())

                // VAD: Calculate energy
                var sum = 0.0
                samples.forEach { sum += it * it }
                lastEnergy = if (samples.isNotEmpty()) sqrt(sum / samples.size) else 0.0

                if (lastEnergy < 0.01) {
                    if (silenceStart == 0L) silenceStart = System.currentTimeMillis()
                    if (System.currentTimeMillis() - silenceStart > vadSilenceThresholdMs) {
                        recorder.stop()
                        whisper.start() // Trigger final transcription if not automatic
                    }
                } else {
                    silenceStart = 0L
                }
            }
        })

        recorder.start()
    }

    private fun correctToFirstPerson(text: String): String {
        if (text.isEmpty()) return ""
        var corrected = text.replace(Regex("\\b(you|he|she|they)\\b", RegexOption.IGNORE_CASE), "I")
            .replace(Regex("\\b(your|his|her|their)\\b", RegexOption.IGNORE_CASE), "my")
        if (!corrected.matches(Regex("I .*", RegexOption.IGNORE_CASE))) {
            corrected = "I " + corrected.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
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
        ByteBuffer.wrap(audioBytes).asShortBuffer().get(shortArray)
        val volume = lastEnergy // Use last energy or calculate

        val arousal = ((avgPitch / 150.0) + (pitchVariance / 50.0) + (volume * 5.0)) / 3.0.coerceIn(0.0, 1.0)

        return Prosody(avgPitch, rate.coerceIn(0.5, 2.0), arousal, volume.coerceIn(0.0, 1.0), pitchVariance)
    }

    private fun speakCorrectedText(text: String, prosody: Prosody) {
        if (text.isEmpty()) return

        // Detect language
        val detectedLanguage = languageDetector.detectLanguageOf(text) ?: Language.ENGLISH
        val langCode = when (detectedLanguage) {
            Language.SPANISH -> "es"
            Language.FRENCH -> "fr"
            else -> "en"
        }

        val offlineTts = ttsMap[langCode] ?: ttsMap["en"] ?: return

        val generatedAudio = offlineTts.generate(text, speed = prosody.rate.toFloat(), speakerId = 0)

        val samples = generatedAudio.samples
        val ttsSampleRate = generatedAudio.sampleRate

        val dispatcher = AudioDispatcherFactory.fromFloatArray(samples, ttsSampleRate, 1024, 512)

        val pitchShiftFactor = (prosody.pitch / 120.0).toFloat()
        val pitchShift = PitchShifter(pitchShiftFactor, ttsSampleRate.toFloat(), 1024, 10)
        dispatcher.addAudioProcessor(pitchShift)

        val gain = prosody.volume * 2.0
        val gainProcessor = GainProcessor(gain)
        dispatcher.addAudioProcessor(gainProcessor)

        val processedBuffer = FloatArray(samples.size * 2)
        var index = 0
        dispatcher.addAudioProcessor(object : AudioProcessor {
            override fun process(audioEvent: AudioEvent): Boolean {
                val buffer = audioEvent.floatBuffer
                buffer.copyInto(processedBuffer, index)
                index += buffer.size
                return true
            }
            override fun processingFinished() {}
        })
        dispatcher.run()

        val processedSamples = processedBuffer.copyOf(index)

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
