      1 -package com.exocortex.neuroacoustic
      2 -
      3 -import android.content.Context
      4 -import android.content.Intent
      5 -import android.content.SharedPreferences
      6 -import android.media.AudioAttributes
      7 -import android.media.AudioFormat
      8 -import android.media.AudioRecord
      9 -import android.media.AudioTrack
     10 -import android.media.MediaRecorder
     11 -import android.media.audiofx.NoiseSuppressor
     12 -import android.util.Log
     13 -import androidx.core.content.ContextCompat
     14 -import androidx.localbroadcastmanager.content.LocalBroadcastManager
     15 -import be.tarsos.dsp.AudioDispatcherFactory
     16 -import be.tarsos.dsp.AudioEvent
     17 -import be.tarsos.dsp.AudioProcessor
     18 -import be.tarsos.dsp.gain.GainProcessor
     19 -import be.tarsos.dsp.pitch.PitchDetectionHandler
     20 -import be.tarsos.dsp.pitch.PitchDetectionResult
     21 -import be.tarsos.dsp.pitch.PitchProcessor
     22 -import be.tarsos.dsp.pitch.PitchShifter
     23 -import com.github.pemistahl.lingua.api.Language
     24 -import com.github.pemistahl.lingua.api.LanguageDetectorBuilder
     25 -import com.k2fsa.sherpa.onnx.OfflineTts
     26 -import com.k2fsa.sherpa.onnx.OfflineTtsConfig
     27 -import org.gkonovalov.android.vad.FrameSize
     28 -import org.gkonovalov.android.vad.Mode
     29 -import org.gkonovalov.android.vad.SampleRate
     30 -import org.gkonovalov.android.vad.VadWebRTC
     31 -import org.json.JSONObject
     32 -import org.kaldi.Model
     33 -import org.kaldi.Recognizer
     34 -import java.io.ByteArrayOutputStream
     35 -import java.io.File
     36 -import java.nio.ByteBuffer
     37 -import kotlin.math.pow
     38 -import kotlin.math.sqrt
     39 -
     40 -class NeuroAcousticMirror(private val context: Context) {
     41 -    private val sampleRate = 16000
     42 -    private val audioBuffer = ByteArrayOutputStream()
     43 -    private val ttsMap = mutableMapOf<String, OfflineTts>()
     44 -    private val languageDetector = LanguageDetectorBuilder.fromLanguages(Language.ENGLISH, Language.SPANISH, Language.F
         RENCH).build()
     45 -    private val sharedPrefs: SharedPreferences = context.getSharedPreferences("vad_prefs", Context.MODE_PRIVATE)
     46 -    private val localBroadcastManager = LocalBroadcastManager.getInstance(context)
     47 -    private val models = mutableMapOf<String, Model>()
     48 -
     49 -    private var vad = VadWebRTC.builder()
     50 -        .setSampleRate(SampleRate.SAMPLE_RATE_16K)
     51 -        .setFrameSize(FrameSize.FRAME_SIZE_320)
     52 -        .setMode(Mode.NORMAL)
     53 -        .setSilenceDurationMs(1200)
     54 -        .setSpeechDurationMs(100)
     55 -        .build()
     56 -
     57 -    private var vadModeIndex = 2
     58 -    private var silenceDurationMs: Int = 1200
     59 -    private var vadThreshold: Float = 0.5f
     60 -    private var noiseSuppressor: NoiseSuppressor? = null
     61 -
     62 -    private var totalFrames = 0
     63 -    private var speechFrames = 0
     64 -    private var silenceStart = 0L
     65 -
     66 -    init {
     67 -        initTts()
     68 -        initVosk()
     69 -        updateVadFromPrefs()
     70 -    }
     71 -
     72 -    private fun initTts() {
     73 -        val modelsList = listOf(
     74 -            "en" to "vits-piper-en_US-amy-medium.onnx",
     75 -            "es" to "vits-piper-es_ES-mls_9972-medium.onnx",
     76 -            "fr" to "vits-piper-fr_FR-upmc-medium.onnx"
     77 -        )
     78 -        modelsList.forEach { (lang, modelName) ->
     79 -            val path = File(context.filesDir, modelName).absolutePath
     80 -            val config = OfflineTtsConfig(model = path, numThreads = 1, debug = false)
     81 -            ttsMap[lang] = OfflineTts(config)
     82 -        }
     83 -    }
     84 -
     85 -    private fun initVosk() {
     86 -        org.kaldi.Vosk.init(context)
     87 -        val voskDirs = listOf(
     88 -            "en" to "vosk-model-small-en-us",
     89 -            "es" to "vosk-model-small-es",
     90 -            "fr" to "vosk-model-small-fr"
     91 -        )
     92 -        voskDirs.forEach { (lang, dir) ->
     93 -            val base = File(context.filesDir, dir)
     94 -            val modelDir = resolveModelDir(base)
     95 -            if (modelDir != null && modelDir.exists()) {
     96 -                models[lang] = Model(modelDir.absolutePath)
     97 -            } else {
     98 -                Log.w("Mirror", "Vosk model missing for $lang at ${base.absolutePath}")
     99 -            }
    100 -        }
    101 -    }
    102 -
    103 -    private fun resolveModelDir(base: File): File? {
    104 -        if (!base.exists()) return null
    105 -        val children = base.listFiles()
    106 -        return if (children != null && children.size == 1 && children[0].isDirectory) {
    107 -            children[0]
    108 -        } else {
    109 -            base
    110 -        }
    111 -    }
    112 -
    113 -    private fun updateVadFromPrefs() {
    114 -        vadModeIndex = sharedPrefs.getInt("vad_mode", 2)
    115 -        silenceDurationMs = sharedPrefs.getInt("silence_ms", 1200)
    116 -        vadThreshold = sharedPrefs.getFloat("vad_threshold", 0.5f)
    117 -        rebuildVad()
    118 -    }
    119 -
    120 -    private fun rebuildVad() {
    121 -        vad.close()
    122 -        val mode = when (vadModeIndex) {
    123 -            0 -> Mode.LOW_BITRATE
    124 -            1 -> Mode.NORMAL
    125 -            3 -> Mode.AGGRESSIVE
    126 -            4 -> Mode.VERY_AGGRESSIVE
    127 -            else -> Mode.NORMAL
    128 -        }
    129 -        vad = VadWebRTC.builder()
    130 -            .setSampleRate(SampleRate.SAMPLE_RATE_16K)
    131 -            .setFrameSize(FrameSize.FRAME_SIZE_320)
    132 -            .setMode(mode)
    133 -            .setSilenceDurationMs(silenceDurationMs)
    134 -            .setSpeechDurationMs(100)
    135 -            .build()
    136 -    }
    137 -
    138 -    fun tuneVAD(mode: Int, silenceMs: Int, threshold: Float) {
    139 -        vadModeIndex = mode
    140 -        silenceDurationMs = silenceMs
    141 -        vadThreshold = threshold.coerceIn(0.1f, 0.9f)
    142 -        rebuildVad()
    143 -    }
    144 -
    145 -    fun listenAndProcess(callback: (String, Prosody) -> Unit) {
    146 -        if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.RECORD_AUDIO) != android.content.pm.
         PackageManager.PERMISSION_GRANTED) {
    147 -            Log.w("Mirror", "Audio permission not granted.")
    148 -            return
    149 -        }
    150 -        if (models.isEmpty()) {
    151 -            Log.w("Mirror", "No Vosk models loaded.")
    152 -            return
    153 -        }
    154 -        resetState()
    155 -
    156 -        val recognizers: Map<String, Recognizer> = models.mapValues { (_, model) ->
    157 -            org.kaldi.Vosk.createRecognizer(model, sampleRate.toFloat())
    158 -        }
    159 -
    160 -        val audioRecord = AudioRecord(
    161 -            MediaRecorder.AudioSource.MIC,
    162 -            sampleRate,
    163 -            AudioFormat.CHANNEL_IN_MONO,
    164 -            AudioFormat.ENCODING_PCM_16BIT,
    165 -            AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
    166 -        )
    167 -
    168 -        if (NoiseSuppressor.isAvailable()) {
    169 -            noiseSuppressor = NoiseSuppressor.create(audioRecord.audioSessionId)
    170 -        }
    171 -
    172 -        audioRecord.startRecording()
    173 -        broadcastStatus("Listening...")
    174 -        val buffer = ShortArray(512)
    175 -
    176 -        while (true) {
    177 -            val read = audioRecord.read(buffer, 0, buffer.size)
    178 -            if (read <= 0) continue
    179 -
    180 -            val frame = buffer.copyOf(read)
    181 -            totalFrames++
    182 -            val frameBytes = ShortArrayToByteArray(frame)
    183 -            val rms = calculateRms(frame)
    184 -            val isSpeech = vad.isSpeech(frameBytes) || rms > (vadThreshold * 0.1)
    185 -
    186 -            if (isSpeech) {
    187 -                speechFrames++
    188 -                silenceStart = 0
    189 -            } else if (silenceStart == 0L && speechFrames > 0) {
    190 -                silenceStart = System.currentTimeMillis()
    191 -            }
    192 -
    193 -            recognizers.values.forEach { it.acceptWaveForm(frame, frame.size) }
    194 -            collectForProsody(frame)
    195 -
    196 -            if (silenceStart > 0 && System.currentTimeMillis() - silenceStart >= silenceDurationMs) {
    197 -                audioRecord.stop()
    198 -                audioRecord.release()
    199 -                noiseSuppressor?.release()
    200 -
    201 -                val best = selectBestResult(recognizers)
    202 -                if (best.first.isNotEmpty()) {
    203 -                    val rawText = best.first
    204 -                    val prosody = extractProsody(audioBuffer.toByteArray(), rawText)
    205 -                    val correctedText = correctToFirstPerson(rawText)
    206 -                    speakCorrectedText(correctedText, prosody)
    207 -                    callback(correctedText, prosody)
    208 -                    logAndBroadcastMetrics()
    209 -                }
    210 -                break
    211 -            }
    212 -        }
    213 -    }
    214 -
    215 -    private fun resetState() {
    216 -        audioBuffer.reset()
    217 -        totalFrames = 0
    218 -        speechFrames = 0
    219 -        vadLatencySum = 0
    220 -        silenceStart = 0
    221 -    }
    222 -
    223 -    private fun ShortArrayToByteArray(samples: ShortArray): ByteArray {
    224 -        val byteArray = ByteArray(samples.size * 2)
    225 -        ByteBuffer.wrap(byteArray).asShortBuffer().put(samples)
    226 -        return byteArray
    227 -    }
    228 -
    229 -    private fun selectBestResult(recognizers: Map<String, Recognizer>): Pair<String, Double> {
    230 -        val results = recognizers.map { (lang, rec) ->
    231 -            val json = try { rec.finalResult() } catch (_: Exception) { rec.result() }
    232 -            val parsed = parseResultWithConfidence(json)
    233 -            Triple(lang, parsed.first, parsed.second)
    234 -        }
    235 -        val best = results.maxByOrNull { it.third } ?: return "" to 0.0
    236 -        return best.second to best.third
    237 -    }
    238 -
    239 -    private fun parseResultWithConfidence(json: String?): Pair<String, Double> {
    240 -        if (json.isNullOrEmpty()) return "" to 0.0
    241 -        val obj = JSONObject(json)
    242 -        val text = obj.optString("text", "")
    243 -        val conf = if (obj.has("result")) {
    244 -            val arr = obj.getJSONArray("result")
    245 -            (0 until arr.length()).map { arr.getJSONObject(it).optDouble("conf", 0.0) }.average()
    246 -        } else 0.0
    247 -        return text to if (conf.isNaN()) 0.0 else conf
    248 -    }
    249 -
    250 -    private fun collectForProsody(samples: ShortArray) {
    251 -        val byteArray = ByteArray(samples.size * 2)
    252 -        ByteBuffer.wrap(byteArray).asShortBuffer().put(samples)
    253 -        audioBuffer.write(byteArray)
    254 -    }
    255 -
    256 -    private fun extractProsody(audioBytes: ByteArray, text: String): Prosody {
    257 -        if (audioBytes.isEmpty()) return Prosody(120.0, 1.0, 0.5, 0.5, 20.0)
    258 -
    259 -        val dispatcher = AudioDispatcherFactory.fromByteArray(audioBytes, sampleRate, 1024, 512)
    260 -        val pitches = mutableListOf<Float>()
    261 -        dispatcher.addAudioProcessor(PitchProcessor(PitchProcessor.PitchEstimationAlgorithm.YIN, sampleRate.toFloat(),
         1024, object : PitchDetectionHandler {
    262 -            override fun handlePitch(result: PitchDetectionResult, event: AudioEvent) {
    263 -                if (result.pitch > 0) pitches.add(result.pitch)
    264 -            }
    265 -        }))
    266 -        dispatcher.run()
    267 -        val avgPitch = if (pitches.isNotEmpty()) pitches.average() else 120.0
    268 -        val pitchVariance = if (pitches.size > 1) pitches.map { (it - avgPitch).pow(2) }.average() else 0.0
    269 -
    270 -        val durationSec = audioBytes.size / (sampleRate * 2).toDouble()
    271 -        val wordCount = text.split("\\s+".toRegex()).size.coerceAtLeast(1)
    272 -        val rate = if (durationSec > 0) (wordCount / durationSec) / 1.5 else 1.0
    273 -
    274 -        val shortArray = ShortArray(audioBytes.size / 2)
    275 -        ByteBuffer.wrap(audioBytes).asShortBuffer().get(shortArray)
    276 -        var sum = 0.0
    277 -        shortArray.forEach { sum += (it / 32768.0).pow(2) }
    278 -        val volume = if (shortArray.isNotEmpty()) sqrt(sum / shortArray.size) else 0.0
    279 -
    280 -        val arousal = ((avgPitch / 150.0) + (pitchVariance / 50.0) + (volume * 5.0)) / 3.0
    281 -        return Prosody(avgPitch, rate.coerceIn(0.5, 2.0), arousal.coerceIn(0.0, 1.0), volume.coerceIn(0.0, 1.0), pitchV
         ariance)
    282 -    }
    283 -
    284 -    private fun calculateRms(frame: ShortArray): Double {
    285 -        var sum = 0.0
    286 -        frame.forEach { sum += (it / 32768.0).pow(2) }
    287 -        return if (frame.isNotEmpty()) sqrt(sum / frame.size) else 0.0
    288 -    }
    289 -
    290 -    private fun correctToFirstPerson(text: String): String {
    291 -        if (text.isEmpty()) return ""
    292 -        var corrected = text.replace(Regex("\\b(you|he|she|they)\\b", RegexOption.IGNORE_CASE), "I")
    293 -            .replace(Regex("\\b(your|his|her|their)\\b", RegexOption.IGNORE_CASE), "my")
    294 -        if (!corrected.matches(Regex("I .*", RegexOption.IGNORE_CASE))) {
    295 -            corrected = "I " + corrected.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
    296 -        }
    297 -        return corrected
    298 -    }
    299 -
    300 -    private fun speakCorrectedText(text: String, prosody: Prosody) {
    301 -        if (text.isEmpty()) return
    302 -        val detectedLanguage = languageDetector.detectLanguageOf(text) ?: Language.ENGLISH
    303 -        val langCode = when (detectedLanguage) {
    304 -            Language.SPANISH -> "es"
    305 -            Language.FRENCH -> "fr"
    306 -            else -> "en"
    307 -        }
    308 -        val offlineTts = ttsMap[langCode] ?: ttsMap["en"] ?: return
    309 -
    310 -        val generatedAudio = offlineTts.generate(text, speed = prosody.rate.toFloat(), speakerId = 0)
    311 -        val samples = generatedAudio.samples
    312 -        val ttsSampleRate = generatedAudio.sampleRate
    313 -
    314 -        val dispatcher = AudioDispatcherFactory.fromFloatArray(samples, ttsSampleRate, 1024, 512)
    315 -        val pitchShiftFactor = (prosody.pitch / 120.0).toFloat()
    316 -        val pitchShift = PitchShifter(pitchShiftFactor, ttsSampleRate.toFloat(), 1024, 10)
    317 -        val gainProcessor = GainProcessor(prosody.volume * 2.0)
    318 -        dispatcher.addAudioProcessor(pitchShift)
    319 -        dispatcher.addAudioProcessor(gainProcessor)
    320 -
    321 -        val processedBuffer = FloatArray(samples.size * 2)
    322 -        var index = 0
    323 -        dispatcher.addAudioProcessor(object : AudioProcessor {
    324 -            override fun process(audioEvent: AudioEvent): Boolean {
    325 -                val buffer = audioEvent.floatBuffer
    326 -                buffer.copyInto(processedBuffer, index)
    327 -                index += buffer.size
    328 -                return true
    329 -            }
    330 -            override fun processingFinished() {}
    331 -        })
    332 -        dispatcher.run()
    333 -        val processedSamples = processedBuffer.copyOf(index)
    334 -        playAudio(processedSamples, ttsSampleRate)
    335 -    }
    336 -
    337 -    private fun playAudio(samples: FloatArray, sampleRate: Int) {
    338 -        val bufferSize = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM
         _FLOAT)
    339 -        val audioTrack = AudioTrack.Builder()
    340 -            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAtt
         ributes.CONTENT_TYPE_SPEECH).build())
    341 -            .setAudioFormat(AudioFormat.Builder().setSampleRate(sampleRate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO
         ).setEncoding(AudioFormat.ENCODING_PCM_FLOAT).build())
    342 -            .setBufferSizeInBytes(bufferSize)
    343 -            .setTransferMode(AudioTrack.MODE_STREAM)
    344 -            .build()
    345 -
    346 -        audioTrack.play()
    347 -        audioTrack.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
    348 -        audioTrack.stop()
    349 -        audioTrack.release()
    350 -    }
    351 -
    352 -    private fun logAndBroadcastMetrics() {
    353 -        val speechRate = if (totalFrames > 0) speechFrames.toFloat() / totalFrames else 0f
    354 -        val metrics = """
    355 -            VAD Performance:
    356 -            Frames: $totalFrames/$speechFrames
    357 -            Speech Rate: ${"%.1f".format(speechRate * 100)}%
    358 -            Threshold: ${"%.2f".format(vadThreshold)}
    359 -            Silence: ${silenceDurationMs}ms
    360 -        """.trimIndent()
    361 -        val intent = Intent("VAD_METRICS_UPDATE").putExtra("metrics", metrics)
    362 -        localBroadcastManager.sendBroadcast(intent)
    363 -    }
    364 -
    365 -    private fun broadcastStatus(status: String) {
    366 -        val intent = Intent("SYSTEM_STATUS_UPDATE").putExtra("status", status)
    367 -        localBroadcastManager.sendBroadcast(intent)
    368 -    }
    369 -
    370 -    fun cleanup() {
    371 -        try {
    372 -            noiseSuppressor?.release()
    373 -        } catch (_: Exception) { }
    374 -        try {
    375 -            vad.close()
    376 -        } catch (_: Exception) { }
    377 -    }
    378 -}

