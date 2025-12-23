package com.exocortex.neuroacoustic

import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.NoiseSuppressor
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import be.tarsos.dsp.AudioDispatcherFactory
import be.tarsos.dsp.AudioEvent
import be.tarsos.dsp.AudioProcessor
import be.tarsos.dsp.gain.GainProcessor
import be.tarsos.dsp.pitch.PitchDetectionHandler
import be.tarsos.dsp.pitch.PitchDetectionResult
import be.tarsos.dsp.pitch.PitchProcessor
import be.tarsos.dsp.pitch.PitchShifter
import com.github.pemistahl.lingua.api.Language
import com.github.pemistahl.lingua.api.LanguageDetectorBuilder
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import org.gkonovalov.android.vad.FrameSize
import org.gkonovalov.android.vad.Mode
import org.gkonovalov.android.vad.SampleRate
import org.gkonovalov.android.vad.VadWebRTC
import org.json.JSONObject
import org.kaldi.Model
import org.kaldi.Recognizer
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.pow
import kotlin.math.sqrt

class NeuroAcousticMirror(private val context: Context) {
    private val sampleRate = 16000
    private val audioBuffer = ByteArrayOutputStream()
    private val ttsMap = mutableMapOf<String, OfflineTts>()
    private val languageDetector = LanguageDetectorBuilder.fromLanguages(Language.ENGLISH, Language.SPANISH, Language.FRENCH).build()
    private val sharedPrefs: SharedPreferences = context.getSharedPreferences("vad_prefs", Context.MODE_PRIVATE)
    private val localBroadcastManager = LocalBroadcastManager.getInstance(context)
    private val models = mutableMapOf<String, Model>()

    private var vad = VadWebRTC.builder()
        .setSampleRate(SampleRate.SAMPLE_RATE_16K)
        .setFrameSize(FrameSize.FRAME_SIZE_320)
        .setMode(Mode.NORMAL)
        .setSilenceDurationMs(1200)
        .setSpeechDurationMs(100)
        .build()

    private var vadModeIndex = 2
    private var silenceDurationMs: Int = 1200
    private var vadThreshold: Float = 0.5f
    private var noiseSuppressor: NoiseSuppressor? = null

    private var totalFrames = 0
    private var speechFrames = 0
    private var silenceStart = 0L

    init {
        initTts()
        initVosk()
        updateVadFromPrefs()
    }

    private fun initTts() {
        val modelsList = listOf(
            "en" to "vits-piper-en_US-amy-medium.onnx",
            "es" to "vits-piper-es_ES-mls_9972-medium.onnx",
            "fr" to "vits-piper-fr_FR-upmc-medium.onnx"
        )
        modelsList.forEach { (lang, modelName) ->
            val path = File(context.filesDir, modelName).absolutePath
            val config = OfflineTtsConfig(model = path, numThreads = 1, debug = false)
            ttsMap[lang] = OfflineTts(config)
        }
    }

    private fun initVosk() {
        org.kaldi.Vosk.init(context)
        val voskDirs = listOf(
            "en" to "vosk-model-small-en-us",
            "es" to "vosk-model-small-es",
            "fr" to "vosk-model-small-fr"
        )
        voskDirs.forEach { (lang, dir) ->
            val base = File(context.filesDir, dir)
            val modelDir = resolveModelDir(base)
            if (modelDir != null && modelDir.exists()) {
                models[lang] = Model(modelDir.absolutePath)
            } else {
                Log.w("Mirror", "Vosk model missing for $lang at ${base.absolutePath}")
            }
        }
    }

    private fun resolveModelDir(base: File): File? {
        if (!base.exists()) return null
        val children = base.listFiles()
        return if (children != null && children.size == 1 && children[0].isDirectory) children[0] else base
    }

    private fun updateVadFromPrefs() {
        vadModeIndex = sharedPrefs.getInt("vad_mode", 2)
        silenceDurationMs = sharedPrefs.getInt("silence_ms", 1200)
        vadThreshold = sharedPrefs.getFloat("vad_threshold", 0.5f)
        rebuildVad()
    }

    private fun rebuildVad() {
        vad.close()
        val mode = when (vadModeIndex) {
            0 -> Mode.LOW_BITRATE
            1 -> Mode.NORMAL
            3 -> Mode.AGGRESSIVE
            4 -> Mode.VERY_AGGRESSIVE
            else -> Mode.NORMAL
        }
        vad = VadWebRTC.builder()
            .setSampleRate(SampleRate.SAMPLE_RATE_16K)
            .setFrameSize(FrameSize.FRAME_SIZE_320)
            .setMode(mode)
            .setSilenceDurationMs(silenceDurationMs)
            .setSpeechDurationMs(100)
            .build()
    }

    fun tuneVAD(mode: Int, silenceMs: Int, threshold: Float) {
        vadModeIndex = mode
        silenceDurationMs = silenceMs
        vadThreshold = threshold.coerceIn(0.1f, 0.9f)
        rebuildVad()
    }

    fun listenAndProcess(callback: (String, Prosody) -> Unit) {
        if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.RECORD_AUDIO) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            Log.w("Mirror", "Audio permission not granted.")
            return
        }
        if (models.isEmpty()) {
            Log.w("Mirror", "No Vosk models loaded.")
            return
        }
        resetState()

        val recognizers: Map<String, Recognizer> = models.mapValues { (_, model) ->
            org.kaldi.Vosk.createRecognizer(model, sampleRate.toFloat())
        }

        val audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        )

        if (NoiseSuppressor.isAvailable()) {
            noiseSuppressor = NoiseSuppressor.create(audioRecord.audioSessionId)
        }

        audioRecord.startRecording()
        broadcastStatus("Listening...")

        // WebRTC VAD expects exact frame sizes. For 16 kHz, 20 ms = 320 samples.
        val frameSize = 320
        val buffer = ShortArray(frameSize)

        while (true) {
            var read = 0
            while (read < frameSize) {
                val r = audioRecord.read(buffer, read, frameSize - read)
                if (r < 0) break
                read += r
            }
            if (read < frameSize) break

            totalFrames++
            val frame = buffer.copyOf(frameSize)
            val frameBytes = ShortArrayToByteArray(frame)
            val rms = calculateRms(frame)
            val isSpeech = vad.isSpeech(frameBytes) || rms > (vadThreshold * 0.1)

            if (isSpeech) {
                speechFrames++
                silenceStart = 0
            } else if (silenceStart == 0L && speechFrames > 0) {
                silenceStart = System.currentTimeMillis()
            }

            recognizers.values.forEach { it.acceptWaveForm(frame, frame.size) }
            collectForProsody(frame)

            if (silenceStart > 0 && System.currentTimeMillis() - silenceStart >= silenceDurationMs) {
                audioRecord.stop()
                audioRecord.release()
                noiseSuppressor?.release()

                val best = selectBestResult(recognizers)
                if (best.first.isNotEmpty()) {
                    val rawText = best.first
                    val prosody = extractProsody(audioBuffer.toByteArray(), rawText)
                    val correctedText = correctToFirstPerson(rawText)
                    speakCorrectedText(correctedText, prosody)
                    callback(correctedText, prosody)
                    logAndBroadcastMetrics()
                }
                break
            }
        }
    }

    private fun resetState() {
        audioBuffer.reset()
        totalFrames = 0
        speechFrames = 0
        silenceStart = 0
    }

    private fun ShortArrayToByteArray(samples: ShortArray): ByteArray {
        val byteArray = ByteArray(samples.size * 2)
        ByteBuffer.wrap(byteArray).asShortBuffer().put(samples)
        return byteArray
    }

    private fun selectBestResult(recognizers: Map<String, Recognizer>): Pair<String, Double> {
        val results = recognizers.map { (_, rec) ->
            val json = try { rec.finalResult() } catch (_: Exception) { rec.result() }
            val parsed = parseResultWithConfidence(json)
            parsed
        }
        val best = results.maxByOrNull { it.second } ?: return "" to 0.0
        return best
    }

    private fun parseResultWithConfidence(json: String?): Pair<String, Double> {
        if (json.isNullOrEmpty()) return "" to 0.0
        val obj = JSONObject(json)
        val text = obj.optString("text", "")
        val conf = if (obj.has("result")) {
            val arr = obj.getJSONArray("result")
            (0 until arr.length()).map { arr.getJSONObject(it).optDouble("conf", 0.0) }.average()
        } else 0.0
        return text to if (conf.isNaN()) 0.0 else conf
    }

    private fun collectForProsody(samples: ShortArray) {
        val byteArray = ByteArray(samples.size * 2)
        ByteBuffer.wrap(byteArray).asShortBuffer().put(samples)
        audioBuffer.write(byteArray)
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
        val wordCount = text.split("\\s+".toRegex()).size.coerceAtLeast(1)
        val rate = if (durationSec > 0) (wordCount / durationSec) / 1.5 else 1.0

        val shortArray = ShortArray(audioBytes.size / 2)
        ByteBuffer.wrap(audioBytes).asShortBuffer().get(shortArray)
        var sum = 0.0
        shortArray.forEach { sum += (it / 32768.0).pow(2) }
        val volume = if (shortArray.isNotEmpty()) sqrt(sum / shortArray.size) else 0.0

        val arousal = ((avgPitch / 150.0) + (pitchVariance / 50.0) + (volume * 5.0)) / 3.0
        return Prosody(avgPitch, rate.coerceIn(0.5, 2.0), arousal.coerceIn(0.0, 1.0), volume.coerceIn(0.0, 1.0), pitchVariance)
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

    private fun speakCorrectedText(text: String, prosody: Prosody) {
        if (text.isEmpty()) return
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
        val gainProcessor = GainProcessor(prosody.volume * 2.0)
        dispatcher.addAudioProcessor(pitchShift)
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

    private fun logAndBroadcastMetrics() {
        val speechRate = if (totalFrames > 0) speechFrames.toFloat() / totalFrames else 0f
        val metrics = """
            VAD Performance:
            Frames: $totalFrames/$speechFrames
            Speech Rate: ${"%.1f".format(speechRate * 100)}%
            Threshold: ${"%.2f".format(vadThreshold)}
            Silence: ${silenceDurationMs}ms
        """.trimIndent()
        val intent = Intent("VAD_METRICS_UPDATE").putExtra("metrics", metrics)
        localBroadcastManager.sendBroadcast(intent)
    }

    private fun broadcastStatus(status: String) {
        val intent = Intent("SYSTEM_STATUS_UPDATE").putExtra("status", status)
        localBroadcastManager.sendBroadcast(intent)
    }

    fun cleanup() {
        try {
            noiseSuppressor?.release()
        } catch (_: Exception) { }
        try {
            vad.close()
        } catch (_: Exception) { }
    }
}
