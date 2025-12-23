private val vadSilenceThresholdMs = 1200L // 1.2s silence for VAD
private val sampleRate = 16000
private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
private var audioRecord: AudioRecord? = null
private var voskModel: Model? = null
private lateinit var tts: TextToSpeech

init {
    // Load Vosk model (assume 'vosk-model-en-us-0.22' in assets)
    val modelPath = File(context.filesDir, "vosk-model-en-us-0.22")
    if (!modelPath.exists()) {
        // Extract from assets (implement extraction logic here)
        // For simplicity, assume pre-extracted
    }
    voskModel = Model(modelPath.absolutePath)

    // Initialize TTS
    tts = TextToSpeech(context) { status ->
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
        }
    }
}

// Listen for user utterance with patient VAD
fun listenAndProcess(callback: (String, Prosody) -> Unit) {
    audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize)
    audioRecord?.startRecording()

    val recognizer = Vosk.createRecognizer(voskModel, sampleRate.toFloat())
    recognizer.setRecognitionListener(object : RecognitionListener {
        override fun onPartialResult(hypothesis: String?) {}
        override fun onResult(hypothesis: String?) {
            val rawText = hypothesis?.let { parseJsonResult(it) } ?: return
            val prosody = extractProsody() // Extract from audio buffer
            val correctedText = correctToFirstPerson(rawText)
            speakCorrectedText(correctedText, prosody)
            callback(correctedText, prosody)
        }
        override fun onFinalResult(hypothesis: String?) {}
        override fun onError(e: Exception?) {}
        override fun onTimeout() {}
    })

    // Simple VAD logic: Read audio until silence > threshold
    val buffer = ShortArray(bufferSize / 2)
    var silenceStart = 0L
    while (true) {
        val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
        if (read > 0) {
            recognizer.acceptWaveForm(buffer, read)
            if (isSilence(buffer)) {
                if (silenceStart == 0L) silenceStart = System.currentTimeMillis()
                if (System.currentTimeMillis() - silenceStart > vadSilenceThresholdMs) {
                    recognizer.result // Trigger result
                    break
                }
            } else {
                silenceStart = 0L
            }
        }
    }

    audioRecord?.stop()
    audioRecord?.release()
}

private fun parseJsonResult(json: String): String {
    // Parse Vosk JSON result (e.g., {"text" : "the actual text"})
    return json.substringAfter("\"text\" : \"").substringBeforeLast("\"")
}

private fun isSilence(buffer: ShortArray): Boolean {
    // Simple energy-based silence detection
    val energy = buffer.map { it.toDouble() * it }.average()
    return energy < 1000.0 // Threshold, tune as needed
}

private fun correctToFirstPerson(text: String): String {
    // Use regex to normalize to first-person (e.g., "you want" -> "I want")
    return text.replace(Regex("\\byou\\b", RegexOption.IGNORE_CASE), "I")
        .replace(Regex("\\byour\\b", RegexOption.IGNORE_CASE), "my")
        // Add more rules as needed
}

private fun extractProsody(): Prosody {
    // Use TarsosDSP to detect pitch and rate
    val dispatcher = AudioDispatcher.fromByteArray(/* audio bytes from buffer */, sampleRate, 1024, 512)
    var pitch = 0.0
    var rate = 1.0 // Default
    dispatcher.addAudioProcessor(PitchProcessor(PitchProcessor.PitchEstimationAlgorithm.YIN, sampleRate.toFloat(), 1024, object : PitchDetectionHandler {
        override fun handlePitch(result: PitchDetectionResult?, event: AudioEvent?) {
            if (result?.pitch != null && result.pitch > 0) pitch = result.pitch.toDouble()
        }
    }))
    dispatcher.run()
    // Estimate rate based on utterance length vs words (simplified)
    rate = 1.0 + Random.nextDouble(-0.2, 0.2) // Placeholder real logic
    return Prosody(pitch, rate)
}

private fun speakCorrectedText(text: String, prosody: Prosody) {
    // Apply prosody to TTS using SSML
    val ssml = "<speak><prosody pitch=\"${prosody.pitch}Hz\" rate=\"${prosody.rate}\">$text</prosody></speak>"
    tts.speak(ssml, TextToSpeech.QUEUE_FLUSH, null, "utteranceId")
    tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
        override fun onStart(utteranceId: String?) {}
        override fun onDone(utteranceId: String?) {}
        override fun onError(utteranceId: String?) {}
    })
}
