private val vadSilenceThresholdMs = 1200L
private val sampleRate = 16000
private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT) * 4
private var audioRecord: AudioRecord? = null
private var voskModel: Model? = null
private val audioBuffer = ByteArrayOutputStream()

init {
    val modelPath = File(context.filesDir, "vosk-model-en-us-0.22")
    voskModel = Model(modelPath.absolutePath)
}

fun listenAndProcess(callback: (String, Prosody) -> Unit) {
    if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return

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
        override fun onError(e: Exception?) { Log.e("Mirror", "Error: ${e?.message}") }
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
            recognizer.acceptWaveForm(shortBuffer, read / 2)
            val energy = calculateRMS(shortBuffer)
            if (energy < 0.01) { // Silence threshold
                if (silenceStart == 0L) silenceStart = System.currentTimeMillis()
                if (System.currentTimeMillis() - silenceStart > vadSilenceThresholdMs && System.currentTimeMillis() - startTime > 2000) { // Min utterance 2s
                    recognizer.result
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
    return json.substringAfter("\"text\" : \"").substringBeforeLast("\"").trim()
}

private fun calculateRMS(buffer: ShortArray): Double {
    var sum = 0.0
    buffer.forEach { sum += (it / 32768.0).pow(2) }
    return sqrt(sum / buffer.size)
}

private fun correctToFirstPerson(text: String): String {
    var corrected = text.replace(Regex("\\byou\\b", RegexOption.IGNORE_CASE), "I")
        .replace(Regex("\\byour\\b", RegexOption.IGNORE_CASE), "my")
        .replace(Regex("\\bhe\\b", RegexOption.IGNORE_CASE), "I")
        .replace(Regex("\\bshe\\b", RegexOption.IGNORE_CASE), "I")
        .replace(Regex("\\bthey\\b", RegexOption.IGNORE_CASE), "I")
        .replace(Regex("\\bwant\\b", RegexOption.IGNORE_CASE), "want") // More rules can be added
    // Ensure first-person perspective
    if (!corrected.startsWith("I ", ignoreCase = true)) {
        corrected = "I " + corrected.lowercase().replaceFirstChar { it.uppercase() }
    }
    return corrected
}

private fun extractProsody(audioBytes: ByteArray, text: String): Prosody {
    // Use TarsosDSP for pitch detection
    val dispatcher = AudioDispatcherFactory.fromByteArray(audioBytes, sampleRate, 1024, 512)
    var pitchSum = 0.0
    var pitchCount = 0
    dispatcher.addAudioProcessor(PitchProcessor(PitchProcessor.PitchEstimationAlgorithm.YIN, sampleRate.toFloat(), 1024, object : PitchDetectionHandler {
        override fun handlePitch(result: PitchDetectionResult, event: AudioEvent) {
            if (result.pitch > 0) {
                pitchSum += result.pitch.toDouble()
                pitchCount++
            }
        }
    }))
    dispatcher.run()
    val avgPitch = if (pitchCount > 0) pitchSum / pitchCount else 120.0 // Default Hz

    // Calculate speaking rate: syllables per second approx (words * 1.5 / duration)
    val durationSec = audioBytes.size / (sampleRate * 2).toDouble() // 16-bit mono
    val wordCount = text.split(" ").size.toDouble()
    val rate = if (durationSec > 0) (wordCount / durationSec) / 2.5 else 1.0 // Normalize to normal rate ~2.5 words/sec

    // Volume: Average RMS
    val shortArray = ShortArray(audioBytes.size / 2)
    java.nio.ByteBuffer.wrap(audioBytes).asShortBuffer().get(shortArray)
    val volume = calculateRMS(shortArray)

    // Arousal: Combine pitch and volume with heuristic
    val arousal = (avgPitch / 150.0 + volume * 10.0) / 2.0 // Normalized 0-1-ish

    return Prosody(avgPitch, rate.coerceIn(0.5, 2.0), arousal.coerceIn(0.0, 1.0), volume.coerceIn(0.0, 1.0))
}

private fun speakCorrectedText(text: String, prosody: Prosody) {
    // Apply prosody using SSML: pitch in Hz, rate as factor, volume as relative
    val pitchStr = "${prosody.pitch}Hz"
    val rateStr = prosody.rate.toString()
    val volumeStr = if (prosody.volume > 0.5) "loud" else if (prosody.volume < 0.3) "soft" else "medium"
    val ssml = "<speak><prosody pitch=\"$pitchStr\" rate=\"$rateStr\" volume=\"$volumeStr\">$text</prosody></speak>"
    tts.speak(ssml, TextToSpeech.QUEUE_FLUSH, null, "exocortex")
    tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
        override fun onStart(utteranceId: String?) {}
        override fun onDone(utteranceId: String?) {}
        override fun onError(utteranceId: String?) { Log.e("TTS", "Error speaking") }
    })
}
