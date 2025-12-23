      1 +package com.exocortex.neuroacoustic
      2 +
      3 +import android.content.Context
      4 +import android.content.Intent
      5 +import android.content.SharedPreferences
      6 +import android.media.AudioAttributes
      7 +import android.media.AudioFormat
      8 +import android.media.AudioRecord
      9 +import android.media.AudioTrack
     10 +import android.media.MediaRecorder
     11 +import android.media.audiofx.NoiseSuppressor
     12 +import android.util.Log
     13 +import androidx.core.content.ContextCompat
     14 +import androidx.localbroadcastmanager.content.LocalBroadcastManager
     15 +import be.tarsos.dsp.AudioDispatcherFactory
     16 +import be.tarsos.dsp.AudioEvent
     17 +import be.tarsos.dsp.AudioProcessor
     18 +import be.tarsos.dsp.gain.GainProcessor
     19 +import be.tarsos.dsp.pitch.PitchDetectionHandler
     20 +import be.tarsos.dsp.pitch.PitchDetectionResult
     21 +import be.tarsos.dsp.pitch.PitchProcessor
     22 +import be.tarsos.dsp.pitch.PitchShifter
     23 +import com.github.pemistahl.lingua.api.Language
     24 +import com.github.pemistahl.lingua.api.LanguageDetectorBuilder
     25 +import com.k2fsa.sherpa.onnx.OfflineTts
     26 +import com.k2fsa.sherpa.onnx.OfflineTtsConfig
     27 +import org.gkonovalov.android.vad.FrameSize
     28 +import org.gkonovalov.android.vad.Mode
     29 +import org.gkonovalov.android.vad.SampleRate
     30 +import org.gkonovalov.android.vad.VadWebRTC
     31 +import org.json.JSONObject
     32 +import org.kaldi.Model
     33 +import org.kaldi.Recognizer
     34 +import java.io.ByteArrayOutputStream
     35 +import java.io.File
     36 +import java.nio.ByteBuffer
     37 +import kotlin.math.pow
     38 +import kotlin.math.sqrt
     39 +
     40 +class NeuroAcousticMirror(private val context: Context) {
     41 +    private val sampleRate = 16000
     42 +    private val audioBuffer = ByteArrayOutputStream()
     43 +    private val ttsMap = mutableMapOf<String, OfflineTts>()
     44 +    private val languageDetector = LanguageDetectorBuilder.fromLanguage
         s(Language.ENGLISH, Language.SPANISH, Language.FRENCH).build()
     45 +    private val sharedPrefs: SharedPreferences = context.getSharedPrefe
         rences("vad_prefs", Context.MODE_PRIVATE)
     46 +    private val localBroadcastManager = LocalBroadcastManager.getInstan
         ce(context)
     47 +    private val models = mutableMapOf<String, Model>()
     48 +
     49 +    private var vad = VadWebRTC.builder()
     50 +        .setSampleRate(SampleRate.SAMPLE_RATE_16K)
     51 +        .setFrameSize(FrameSize.FRAME_SIZE_320)
     52 +        .setMode(Mode.NORMAL)
     53 +        .setSilenceDurationMs(1200)
     54 +        .setSpeechDurationMs(100)
     55 +        .build()
     56 +
     57 +    private var vadModeIndex = 2
     58 +    private var silenceDurationMs: Int = 1200
     59 +    private var vadThreshold: Float = 0.5f
     60 +    private var noiseSuppressor: NoiseSuppressor? = null
     61 +
     62 +    private var totalFrames = 0
     63 +    private var speechFrames = 0
     64 +    private var silenceStart = 0L
     65 +
     66 +    init {
     67 +        initTts()
     68 +        initVosk()
     69 +        updateVadFromPrefs()
     70 +    }
     71 +
     72 +    private fun initTts() {
     73 +        val modelsList = listOf(
     74 +            "en" to "vits-piper-en_US-amy-medium.onnx",
     75 +            "es" to "vits-piper-es_ES-mls_9972-medium.onnx",
     76 +            "fr" to "vits-piper-fr_FR-upmc-medium.onnx"
     77 +        )
     78 +        modelsList.forEach { (lang, modelName) ->
     79 +            val path = File(context.filesDir, modelName).absolutePath
     80 +            val config = OfflineTtsConfig(model = path, numThreads = 1,
          debug = false)
     81 +            ttsMap[lang] = OfflineTts(config)
     82 +        }
     83 +    }
     84 +
     85 +    private fun initVosk() {
     86 +        org.kaldi.Vosk.init(context)
     87 +        val voskDirs = listOf(
     88 +            "en" to "vosk-model-small-en-us",
     89 +            "es" to "vosk-model-small-es",
     90 +            "fr" to "vosk-model-small-fr"
     91 +        )
     92 +        voskDirs.forEach { (lang, dir) ->
     93 +            val path = File(context.filesDir, dir)
     94 +            if (path.exists()) {
     95 +                models[lang] = Model(path.absolutePath)
     96 +            } else {
     97 +                Log.w("Mirror", "Vosk model missing for $lang at ${path
         .absolutePath}")
     98 +            }
     99 +        }
    100 +    }
    101 +
    102 +    private fun updateVadFromPrefs() {
    103 +        vadModeIndex = sharedPrefs.getInt("vad_mode", 2)
    104 +        silenceDurationMs = sharedPrefs.getInt("silence_ms", 1200)
    105 +        vadThreshold = sharedPrefs.getFloat("vad_threshold", 0.5f)
    106 +        rebuildVad()
    107 +    }
    108 +
    109 +    private fun rebuildVad() {
    110 +        vad.close()
    111 +        val mode = when (vadModeIndex) {
    112 +            0 -> Mode.LOW_BITRATE
    113 +            1 -> Mode.NORMAL
    114 +            3 -> Mode.AGGRESSIVE
    115 +            4 -> Mode.VERY_AGGRESSIVE
    116 +            else -> Mode.NORMAL
    117 +        }
    118 +        vad = VadWebRTC.builder()
    119 +            .setSampleRate(SampleRate.SAMPLE_RATE_16K)
    120 +            .setFrameSize(FrameSize.FRAME_SIZE_320)
    121 +            .setMode(mode)
    122 +            .setSilenceDurationMs(silenceDurationMs)
    123 +            .setSpeechDurationMs(100)
    124 +            .build()
    125 +    }
    126 +
    127 +    fun tuneVAD(mode: Int, silenceMs: Int, threshold: Float) {
    128 +        vadModeIndex = mode
    129 +        silenceDurationMs = silenceMs
    130 +        vadThreshold = threshold.coerceIn(0.1f, 0.9f)
    131 +        rebuildVad()
    132 +    }
    133 +
    134 +    fun listenAndProcess(callback: (String, Prosody) -> Unit) {
    135 +        if (ContextCompat.checkSelfPermission(context, android.Manifest
         .permission.RECORD_AUDIO) != android.content.pm.PackageManager.PERMISSI
         ON_GRANTED) {
    136 +            Log.w("Mirror", "Audio permission not granted.")
    137 +            return
    138 +        }
    139 +        resetState()
    140 +
    141 +        val recognizers: Map<String, Recognizer> = models.mapValues { (
         _, model) ->
    142 +            org.kaldi.Vosk.createRecognizer(model, sampleRate.toFloat()
         )
    143 +        }
    144 +
    145 +        val audioRecord = AudioRecord(
    146 +            MediaRecorder.AudioSource.MIC,
    147 +            sampleRate,
    148 +            AudioFormat.CHANNEL_IN_MONO,
    149 +            AudioFormat.ENCODING_PCM_16BIT,
    150 +            AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNE
         L_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
    151 +        )
    152 +
    153 +        if (NoiseSuppressor.isAvailable()) {
    154 +            noiseSuppressor = NoiseSuppressor.create(audioRecord.audioS
         essionId)
    155 +        }
    156 +
    157 +        audioRecord.startRecording()
    158 +        broadcastStatus("Listening...")
    159 +        val buffer = ShortArray(512)
    160 +
    161 +        while (true) {
    162 +            val read = audioRecord.read(buffer, 0, buffer.size)
    163 +            if (read <= 0) continue
    164 +
    165 +            val frame = buffer.copyOf(read)
    166 +            totalFrames++
    167 +            val frameBytes = ShortArrayToByteArray(frame)
    168 +            val isSpeech = vad.isSpeech(frameBytes)
    169 +
    170 +            if (isSpeech) {
    171 +                speechFrames++
    172 +                silenceStart = 0
    173 +            } else if (silenceStart == 0L && speechFrames > 0) {
    174 +                silenceStart = System.currentTimeMillis()
    175 +            }
    176 +
    177 +            recognizers.values.forEach { it.acceptWaveForm(frame, frame
         .size) }
    178 +            collectForProsody(frame)
    179 +
    180 +            if (silenceStart > 0 && System.currentTimeMillis() - silenc
         eStart >= silenceDurationMs) {
    181 +                audioRecord.stop()
    182 +                audioRecord.release()
    183 +                noiseSuppressor?.release()
    184 +
    185 +                val best = selectBestResult(recognizers)
    186 +                if (best.first.isNotEmpty()) {
    187 +                    val rawText = best.first
    188 +                    val prosody = extractProsody(audioBuffer.toByteArra
         y(), rawText)
    189 +                    val correctedText = correctToFirstPerson(rawText)
    190 +                    speakCorrectedText(correctedText, prosody)
    191 +                    callback(correctedText, prosody)
    192 +                    logAndBroadcastMetrics()
    193 +                }
    194 +                break
    195 +            }
    196 +        }
    197 +    }
    198 +
    199 +    private fun resetState() {
    200 +        audioBuffer.reset()
    201 +        totalFrames = 0
    202 +        speechFrames = 0
    203 +        vadLatencySum = 0
    204 +        silenceStart = 0
    205 +    }
    206 +
    207 +    private fun ShortArrayToByteArray(samples: ShortArray): ByteArray {
    208 +        val byteArray = ByteArray(samples.size * 2)
    209 +        ByteBuffer.wrap(byteArray).asShortBuffer().put(samples)
    210 +        return byteArray
    211 +    }
    212 +
    213 +    private fun selectBestResult(recognizers: Map<String, Recognizer>):
          Pair<String, Double> {
    214 +        val results = recognizers.map { (lang, rec) ->
    215 +            val json = rec.finalResult ?: rec.result
    216 +            val parsed = parseResultWithConfidence(json)
    217 +            Triple(lang, parsed.first, parsed.second)
    218 +        }
    219 +        val best = results.maxByOrNull { it.third } ?: return "" to 0.0
    220 +        return best.second to best.third
    221 +    }
    222 +
    223 +    private fun parseResultWithConfidence(json: String?): Pair<String,
         Double> {
    224 +        if (json.isNullOrEmpty()) return "" to 0.0
    225 +        val obj = JSONObject(json)
    226 +        val text = obj.optString("text", "")
    227 +        val conf = if (obj.has("result")) {
    228 +            val arr = obj.getJSONArray("result")
    229 +            (0 until arr.length()).map { arr.getJSONObject(it).optDoubl
         e("conf", 0.0) }.average()
    230 +        } else 0.0
    231 +        return text to if (conf.isNaN()) 0.0 else conf
    232 +    }
    233 +
    234 +    private fun collectForProsody(samples: ShortArray) {
    235 +        val byteArray = ByteArray(samples.size * 2)
    236 +        ByteBuffer.wrap(byteArray).asShortBuffer().put(samples)
    237 +        audioBuffer.write(byteArray)
    238 +    }
    239 +
    240 +    private fun extractProsody(audioBytes: ByteArray, text: String): Pr
         osody {
    241 +        if (audioBytes.isEmpty()) return Prosody(120.0, 1.0, 0.5, 0.5,
         20.0)
    242 +
    243 +        val dispatcher = AudioDispatcherFactory.fromByteArray(audioByte
         s, sampleRate, 1024, 512)
    244 +        val pitches = mutableListOf<Float>()
    245 +        dispatcher.addAudioProcessor(PitchProcessor(PitchProcessor.Pitc
         hEstimationAlgorithm.YIN, sampleRate.toFloat(), 1024, object : PitchDet
         ectionHandler {
    246 +            override fun handlePitch(result: PitchDetectionResult, even
         t: AudioEvent) {
    247 +                if (result.pitch > 0) pitches.add(result.pitch)
    248 +            }
    249 +        }))
    250 +        dispatcher.run()
    251 +        val avgPitch = if (pitches.isNotEmpty()) pitches.average() else
          120.0
    252 +        val pitchVariance = if (pitches.size > 1) pitches.map { (it - a
         vgPitch).pow(2) }.average() else 0.0
    253 +
    254 +        val durationSec = audioBytes.size / (sampleRate * 2).toDouble()
    255 +        val wordCount = text.split("\\s+".toRegex()).size.coerceAtLeast
         (1)
    256 +        val rate = if (durationSec > 0) (wordCount / durationSec) / 1.5
          else 1.0
    257 +
    258 +        val shortArray = ShortArray(audioBytes.size / 2)
    259 +        ByteBuffer.wrap(audioBytes).asShortBuffer().get(shortArray)
    260 +        var sum = 0.0
    261 +        shortArray.forEach { sum += (it / 32768.0).pow(2) }
    262 +        val volume = if (shortArray.isNotEmpty()) sqrt(sum / shortArray
         .size) else 0.0
    263 +
    264 +        val arousal = ((avgPitch / 150.0) + (pitchVariance / 50.0) + (v
         olume * 5.0)) / 3.0
    265 +        return Prosody(avgPitch, rate.coerceIn(0.5, 2.0), arousal.coerc
         eIn(0.0, 1.0), volume.coerceIn(0.0, 1.0), pitchVariance)
    266 +    }
    267 +
    268 +    private fun correctToFirstPerson(text: String): String {
    269 +        if (text.isEmpty()) return ""
    270 +        var corrected = text.replace(Regex("\\b(you|he|she|they)\\b", R
         egexOption.IGNORE_CASE), "I")
    271 +            .replace(Regex("\\b(your|his|her|their)\\b", RegexOption.IG
         NORE_CASE), "my")
    272 +        if (!corrected.matches(Regex("I .*", RegexOption.IGNORE_CASE)))
          {
    273 +            corrected = "I " + corrected.replaceFirstChar { if (it.isLo
         werCase()) it.titlecase() else it.toString() }
    274 +        }
    275 +        return corrected
    276 +    }
    277 +
    278 +    private fun speakCorrectedText(text: String, prosody: Prosody) {
    279 +        if (text.isEmpty()) return
    280 +        val detectedLanguage = languageDetector.detectLanguageOf(text)
         ?: Language.ENGLISH
    281 +        val langCode = when (detectedLanguage) {
    282 +            Language.SPANISH -> "es"
    283 +            Language.FRENCH -> "fr"
    284 +            else -> "en"
    285 +        }
    286 +        val offlineTts = ttsMap[langCode] ?: ttsMap["en"] ?: return
    287 +
    288 +        val generatedAudio = offlineTts.generate(text, speed = prosody.
         rate.toFloat(), speakerId = 0)
    289 +        val samples = generatedAudio.samples
    290 +        val ttsSampleRate = generatedAudio.sampleRate
    291 +
    292 +        val dispatcher = AudioDispatcherFactory.fromFloatArray(samples,
          ttsSampleRate, 1024, 512)
    293 +        val pitchShiftFactor = (prosody.pitch / 120.0).toFloat()
    294 +        val pitchShift = PitchShifter(pitchShiftFactor, ttsSampleRate.t
         oFloat(), 1024, 10)
    295 +        val gainProcessor = GainProcessor(prosody.volume * 2.0)
    296 +        dispatcher.addAudioProcessor(pitchShift)
    297 +        dispatcher.addAudioProcessor(gainProcessor)
    298 +
    299 +        val processedBuffer = FloatArray(samples.size * 2)
    300 +        var index = 0
    301 +        dispatcher.addAudioProcessor(object : AudioProcessor {
    302 +            override fun process(audioEvent: AudioEvent): Boolean {
    303 +                val buffer = audioEvent.floatBuffer
    304 +                buffer.copyInto(processedBuffer, index)
    305 +                index += buffer.size
    306 +                return true
    307 +            }
    308 +            override fun processingFinished() {}
    309 +        })
    310 +        dispatcher.run()
    311 +        val processedSamples = processedBuffer.copyOf(index)
    312 +        playAudio(processedSamples, ttsSampleRate)
    313 +    }
    314 +
    315 +    private fun playAudio(samples: FloatArray, sampleRate: Int) {
    316 +        val bufferSize = AudioTrack.getMinBufferSize(sampleRate, AudioF
         ormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
    317 +        val audioTrack = AudioTrack.Builder()
    318 +            .setAudioAttributes(AudioAttributes.Builder().setUsage(Audi
         oAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SP
         EECH).build())
    319 +            .setAudioFormat(AudioFormat.Builder().setSampleRate(sampleR
         ate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setEncoding(AudioForm
         at.ENCODING_PCM_FLOAT).build())
    320 +            .setBufferSizeInBytes(bufferSize)
    321 +            .setTransferMode(AudioTrack.MODE_STREAM)
    322 +            .build()
    323 +
    324 +        audioTrack.play()
    325 +        audioTrack.write(samples, 0, samples.size, AudioTrack.WRITE_BLO
         CKING)
    326 +        audioTrack.stop()
    327 +        audioTrack.release()
    328 +    }
    329 +
    330 +    private fun logAndBroadcastMetrics() {
    331 +        val speechRate = if (totalFrames > 0) speechFrames.toFloat() /
         totalFrames else 0f
    332 +        val metrics = """
    333 +            VAD Performance:
    334 +            Frames: $totalFrames/$speechFrames
    335 +            Speech Rate: ${"%.1f".format(speechRate * 100)}%
    336 +            Threshold: ${"%.2f".format(vadThreshold)}
    337 +            Silence: ${silenceDurationMs}ms
    338 +        """.trimIndent()
    339 +        val intent = Intent("VAD_METRICS_UPDATE").putExtra("metrics", m
         etrics)
    340 +        localBroadcastManager.sendBroadcast(intent)
    341 +    }
    342 +
    343 +    private fun broadcastStatus(status: String) {
    344 +        val intent = Intent("SYSTEM_STATUS_UPDATE").putExtra("status",
         status)
    345 +        localBroadcastManager.sendBroadcast(intent)
    346 +    }
    347 +
    348 +    fun cleanup() {
    349 +        try {
    350 +            noiseSuppressor?.release()
    351 +        } catch (_: Exception) { }
    352 +        try {
    353 +            vad.close()
    354 +        } catch (_: Exception) { }
    355 +    }
    356 +}

