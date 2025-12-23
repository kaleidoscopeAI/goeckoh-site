      1 -package com.exocortex.neuroacoustic
      2 -
      3 -import android.content.Context
      4 -import android.content.Intent
      5 -import android.content.SharedPreferences
      6 -import android.content.Context
      7 -import android.content.Intent
      8 -import android.content.SharedPreferences
      9 -import android.media.*
     10 -import android.util.Log
     11 -import androidx.core.content.ContextCompat
     12 -import androidx.localbroadcastmanager.content.LocalBroadcastManager
     13 -import be.tarsos.dsp.AudioDispatcherFactory
     14 -import be.tarsos.dsp.AudioEvent
     15 -import be.tarsos.dsp.AudioProcessor
     16 -import be.tarsos.dsp.gain.GainProcessor
     17 -import be.tarsos.dsp.pitch.PitchDetectionHandler
     18 -import be.tarsos.dsp.pitch.PitchDetectionResult
     19 -import be.tarsos.dsp.pitch.PitchProcessor
     20 -import be.tarsos.dsp.pitch.PitchShifter
     21 -import com.github.pemistahl.lingua.api.Language
     22 -import com.github.pemistahl.lingua.api.LanguageDetectorBuilder
     23 -import com.k2fsa.sherpa.onnx.OfflineTts
     24 -import com.k2fsa.sherpa.onnx.OfflineTtsConfig
     25 -import org.gkonovalov.android.vad.FrameSize
     26 -import org.gkonovalov.android.vad.Mode
     27 -import org.gkonovalov.android.vad.SampleRate
     28 -import org.gkonovalov.android.vad.VadWebRTC
     29 -import org.json.JSONObject
     30 -import org.kaldi.Model
     31 -import org.kaldi.Recognizer
     32 -import java.io.ByteArrayOutputStream
     33 -import java.io.File
     34 -import java.nio.ByteBuffer
     35 -import kotlin.math.pow
     36 -import kotlin.math.sqrt
     37 -
     38 -class NeuroAcousticMirror(private val context: Context) {
     39 -    private val sampleRate = 16000
     40 -    private val audioBuffer = ByteArrayOutputStream()
     41 -    private val ttsMap = mutableMapOf<String, OfflineTts>()
     42 -    private val languageDetector = LanguageDetectorBuilder.fromLanguage
         s(Language.ENGLISH, Language.SPANISH, Language.FRENCH).build()
     43 -    private val sharedPrefs: SharedPreferences = context.getSharedPrefe
         rences("vad_prefs", Context.MODE_PRIVATE)
     44 -    private val localBroadcastManager = LocalBroadcastManager.getInstan
         ce(context)
     45 -
     46 -    private val models = mutableMapOf<String, Model>()
     47 -    private var vad = VadWebRTC.builder()
     48 -        .setSampleRate(SampleRate.SAMPLE_RATE_16K)
     49 -        .setFrameSize(FrameSize.FRAME_SIZE_320)
     50 -        .setMode(Mode.NORMAL)
     51 -        .setSilenceDurationMs(1200)
     52 -        .setSpeechDurationMs(100)
     53 -        .build()
     54 -
     55 -    private var vadModeIndex = 2
     56 -    private var silenceDurationMs: Int = 1200
     57 -    private var vadThreshold: Float = 0.5f
     58 -    private var noiseSuppressor: NoiseSuppressor? = null
     59 -
     60 -    private var totalFrames = 0
     61 -    private var speechFrames = 0
     62 -    private var vadLatencySum = 0L
     63 -
     64 -    init {
     65 -        initTts()
     66 -        initVosk()
     67 -        updateVadFromPrefs()
     68 -    }
     69 -
     70 -    private fun initTts() {
     71 -        val modelsList = listOf(
     72 -            "en" to "vits-piper-en_US-amy-medium.onnx",
     73 -            "es" to "vits-piper-es_ES-mls_9972-medium.onnx",
     74 -            "fr" to "vits-piper-fr_FR-upmc-medium.onnx"
     75 -        )
     76 -        modelsList.forEach { (lang, modelName) ->
     77 -            val path = File(context.filesDir, modelName).absolutePath
     78 -            val config = OfflineTtsConfig(model = path, numThreads = 1,
          debug = false)
     79 -            ttsMap[lang] = OfflineTts(config)
     80 -        }
     81 -    }
     82 -
     83 -    private fun initVosk() {
     84 -        org.kaldi.Vosk.init(context)
     85 -        val voskDirs = listOf(
     86 -            "en" to "vosk-model-small-en-us",
     87 -            "es" to "vosk-model-small-es",
     88 -            "fr" to "vosk-model-small-fr"
     89 -        )
     90 -        voskDirs.forEach { (lang, dir) ->
     91 -            val path = File(context.filesDir, dir)
     92 -            if (path.exists()) {
     93 -                models[lang] = Model(path.absolutePath)
     94 -            } else {
     95 -                Log.w("Mirror", "Vosk model missing for $lang at ${path
         .absolutePath}")
     96 -            }
     97 -        }
     98 -    }
     99 -
    100 -    private fun updateVadFromPrefs() {
    101 -        vadModeIndex = sharedPrefs.getInt("vad_mode", 2)
    102 -        silenceDurationMs = sharedPrefs.getInt("silence_ms", 1200)
    103 -        vadThreshold = sharedPrefs.getFloat("vad_threshold", 0.5f)
    104 -        rebuildVad()
    105 -    }
    106 -
    107 -    private fun rebuildVad() {
    108 -        vad.close()
    109 -        val mode = when (vadModeIndex) {
    110 -            0 -> Mode.LOW_BITRATE
    111 -            1 -> Mode.NORMAL
    112 -            3 -> Mode.AGGRESSIVE
    113 -            4 -> Mode.VERY_AGGRESSIVE
    114 -            else -> Mode.NORMAL
    115 -        }
    116 -        vad = VadWebRTC.builder()
    117 -            .setSampleRate(SampleRate.SAMPLE_RATE_16K)
    118 -            .setFrameSize(FrameSize.FRAME_SIZE_320)
    119 -            .setMode(mode)
    120 -            .setSilenceDurationMs(silenceDurationMs)
    121 -            .setSpeechDurationMs(100)
    122 -            .build()
    123 -    }
    124 -
    125 -    fun tuneVAD(mode: Int, silenceMs: Int, threshold: Float) {
    126 -        vadModeIndex = mode
    127 -        silenceDurationMs = silenceMs
    128 -        vadThreshold = threshold.coerceIn(0.1f, 0.9f)
    129 -        rebuildVad()
    130 -    }
    131 -
    132 -    fun listenAndProcess(callback: (String, Prosody) -> Unit) {
    133 -        if (ContextCompat.checkSelfPermission(context, android.Manifest
         .permission.RECORD_AUDIO) != android.content.pm.PackageManager.PERMISSI
         ON_GRANTED) {
    134 -            Log.w("Mirror", "Audio permission not granted.")
    135 -            return
    136 -        }
    137 -        resetState()
    138 -        setupWhisperListener(callback)
    139 -        setupRecorderListener()
    140 -
    141 -        try {
    142 -            recorder.start()
    143 -            broadcastStatus("Listening…")
    144 -        } catch (e: Exception) {
    145 -            broadcastStatus("Recorder error: ${e.message}")
    146 -        }
    147 -    }
    148 -
    149 -    private fun resetState() {
    150 -        audioBuffer.reset()
    151 -        partialText = ""
    152 -        silenceStart = 0
    153 -        isSpeechDetected = false
    154 -        totalFrames = 0
    155 -        speechFrames = 0
    156 -        vadLatencySum = 0
    157 -        lastMetricsUpdate = System.currentTimeMillis()
    158 -    }
    159 -
    160 -    private fun setupWhisperListener(callback: (String, Prosody) -> Uni
         t) {
    161 -        whisper.setAction(Whisper.ACTION_TRANSCRIBE)
    162 -        whisper.setListener(object : IWhisperListener {
    163 -            override fun onUpdateReceived(message: String) {
    164 -                partialText = message.trim()
    165 -            }
    166 -
    167 -            override fun onResultReceived(result: String) {
    168 -                val rawText = result.trim()
    169 -                if (rawText.isEmpty()) return
    170 -                val prosody = extractProsody(audioBuffer.toByteArray(),
          rawText)
    171 -                val correctedText = correctToFirstPerson(rawText)
    172 -                speakCorrectedText(correctedText, prosody)
    173 -                callback(correctedText, prosody)
    174 -                broadcastStatus("Processed: ${correctedText.take(48)}")
    175 -            }
    176 -        })
    177 -    }
    178 -
    179 -    private fun setupRecorderListener() {
    180 -        recorder.setListener(object : IRecorderListener {
    181 -            override fun onUpdateReceived(message: String) {
    182 -                // No-op
    183 -            }
    184 -
    185 -            override fun onDataReceived(samples: FloatArray) {
    186 -                processChunk(samples)
    187 -            }
    188 -        })
    189 -    }
    190 -
    191 -    private fun processChunk(samples: FloatArray) {
    192 -        totalFrames++
    193 -        val shortArray = FloatArrayToShortArray(samples)
    194 -        val cleaned = if (useNoiseSuppression && nsWrapper != null) {
    195 -            try {
    196 -                nsWrapper!!.process(shortArray)
    197 -            } catch (e: Exception) {
    198 -                Log.w("Mirror", "Noise suppression failed, using raw au
         dio: ${e.message}")
    199 -                shortArray
    200 -            }
    201 -        } else shortArray
    202 -
    203 -        forwardToWhisper(cleaned)
    204 -        processVad(cleaned)
    205 -        collectForProsody(cleaned)
    206 -    }
    207 -
    208 -    private fun FloatArrayToShortArray(samples: FloatArray): ShortArray
          {
    209 -        return ShortArray(samples.size) { i ->
    210 -            (samples[i] * Short.MAX_VALUE).toInt().coerceIn(Short.MIN_V
         ALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
    211 -        }
    212 -    }
    213 -
    214 -    private fun forwardToWhisper(samples: ShortArray) {
    215 -        val floatSamples = FloatArray(samples.size) { i -> samples[i].t
         oFloat() / Short.MAX_VALUE }
    216 -        whisper.writeBuffer(floatSamples)
    217 -    }
    218 -
    219 -    private fun processVad(samples: ShortArray) {
    220 -        val frameSize = 512
    221 -        var offset = 0
    222 -        while (offset + frameSize <= samples.size) {
    223 -            val frame = samples.copyOfRange(offset, offset + frameSize)
    224 -            val start = System.currentTimeMillis()
    225 -            val speechProb = sileroVad.process(frame)
    226 -            vadLatencySum += System.currentTimeMillis() - start
    227 -            val isSpeech = speechProb > vadThreshold
    228 -            if (isSpeech) {
    229 -                speechFrames++
    230 -                isSpeechDetected = true
    231 -                silenceStart = 0
    232 -            } else if (isSpeechDetected) {
    233 -                if (silenceStart == 0L) silenceStart = System.currentTi
         meMillis()
    234 -                if (System.currentTimeMillis() - silenceStart >= silenc
         eDurationMs) {
    235 -                    recorder.stop()
    236 -                    whisper.start()
    237 -                    logAndBroadcastMetrics()
    238 -                }
    239 -            }
    240 -            offset += frameSize
    241 -        }
    242 -
    243 -        if (System.currentTimeMillis() - lastMetricsUpdate > 1000) {
    244 -            logAndBroadcastMetrics()
    245 -            lastMetricsUpdate = System.currentTimeMillis()
    246 -        }
    247 -    }
    248 -
    249 -    private fun collectForProsody(samples: ShortArray) {
    250 -        val byteArray = ByteArray(samples.size * 2)
    251 -        ByteBuffer.wrap(byteArray).asShortBuffer().put(samples)
    252 -        audioBuffer.write(byteArray)
    253 -    }
    254 -
    255 -    private fun extractProsody(audioBytes: ByteArray, text: String): Pr
         osody {
    256 -        if (audioBytes.isEmpty()) return Prosody(120.0, 1.0, 0.5, 0.5,
         20.0)
    257 -
    258 -        val dispatcher = AudioDispatcherFactory.fromByteArray(audioByte
         s, sampleRate, 1024, 512)
    259 -        val pitches = mutableListOf<Float>()
    260 -        dispatcher.addAudioProcessor(PitchProcessor(PitchProcessor.Pitc
         hEstimationAlgorithm.YIN, sampleRate.toFloat(), 1024, object : PitchDet
         ectionHandler {
    261 -            override fun handlePitch(result: PitchDetectionResult, even
         t: AudioEvent) {
    262 -                if (result.pitch > 0) pitches.add(result.pitch)
    263 -            }
    264 -        }))
    265 -        dispatcher.run()
    266 -        val avgPitch = if (pitches.isNotEmpty()) pitches.average() else
          120.0
    267 -        val pitchVariance = if (pitches.size > 1) pitches.map { (it - a
         vgPitch).pow(2) }.average() else 0.0
    268 -
    269 -        val durationSec = audioBytes.size / (sampleRate * 2).toDouble()
    270 -        val wordCount = text.split("\\s+".toRegex()).size.coerceAtLeast
         (1)
    271 -        val rate = if (durationSec > 0) (wordCount / durationSec) / 1.5
          else 1.0
    272 -
    273 -        val shortArray = ShortArray(audioBytes.size / 2)
    274 -        ByteBuffer.wrap(audioBytes).asShortBuffer().get(shortArray)
    275 -        var sum = 0.0
    276 -        shortArray.forEach { sum += (it / 32768.0).pow(2) }
    277 -        val volume = if (shortArray.isNotEmpty()) sqrt(sum / shortArray
         .size) else 0.0
    278 -
    279 -        val arousal = ((avgPitch / 150.0) + (pitchVariance / 50.0) + (v
         olume * 5.0)) / 3.0
    280 -        return Prosody(avgPitch, rate.coerceIn(0.5, 2.0), arousal.coerc
         eIn(0.0, 1.0), volume.coerceIn(0.0, 1.0), pitchVariance)
    281 -    }
    282 -
    283 -    private fun correctToFirstPerson(text: String): String {
    284 -        if (text.isEmpty()) return ""
    285 -        var corrected = text.replace(Regex("\\b(you|he|she|they)\\b", R
         egexOption.IGNORE_CASE), "I")
    286 -            .replace(Regex("\\b(your|his|her|their)\\b", RegexOption.IG
         NORE_CASE), "my")
    287 -        if (!corrected.matches(Regex("I .*", RegexOption.IGNORE_CASE)))
          {
    288 -            corrected = "I " + corrected.replaceFirstChar { if (it.isLo
         werCase()) it.titlecase() else it.toString() }
    289 -        }
    290 -        return corrected
    291 -    }
    292 -
    293 -    private fun speakCorrectedText(text: String, prosody: Prosody) {
    294 -        if (text.isEmpty()) return
    295 -        val detectedLanguage = languageDetector.detectLanguageOf(text)
         ?: Language.ENGLISH
    296 -        val langCode = when (detectedLanguage) {
    297 -            Language.SPANISH -> "es"
    298 -            Language.FRENCH -> "fr"
    299 -            else -> "en"
    300 -        }
    301 -        val offlineTts = ttsMap[langCode] ?: ttsMap["en"] ?: return
    302 -
    303 -        val generatedAudio = offlineTts.generate(text, speed = prosody.
         rate.toFloat(), speakerId = 0)
    304 -        val samples = generatedAudio.samples
    305 -        val ttsSampleRate = generatedAudio.sampleRate
    306 -
    307 -        val dispatcher = AudioDispatcherFactory.fromFloatArray(samples,
          ttsSampleRate, 1024, 512)
    308 -        val pitchShiftFactor = (prosody.pitch / 120.0).toFloat()
    309 -        val pitchShift = PitchShifter(pitchShiftFactor, ttsSampleRate.t
         oFloat(), 1024, 10)
    310 -        val gainProcessor = GainProcessor(prosody.volume * 2.0)
    311 -        dispatcher.addAudioProcessor(pitchShift)
    312 -        dispatcher.addAudioProcessor(gainProcessor)
    313 -
    314 -        val processedBuffer = FloatArray(samples.size * 2)
    315 -        var index = 0
    316 -        dispatcher.addAudioProcessor(object : AudioProcessor {
    317 -            override fun process(audioEvent: AudioEvent): Boolean {
    318 -                val buffer = audioEvent.floatBuffer
    319 -                buffer.copyInto(processedBuffer, index)
    320 -                index += buffer.size
    321 -                return true
    322 -            }
    323 -            override fun processingFinished() {}
    324 -        })
    325 -        dispatcher.run()
    326 -        val processedSamples = processedBuffer.copyOf(index)
    327 -        playAudio(processedSamples, ttsSampleRate)
    328 -    }
    329 -
    330 -    private fun playAudio(samples: FloatArray, sampleRate: Int) {
    331 -        val bufferSize = AudioTrack.getMinBufferSize(sampleRate, AudioF
         ormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
    332 -        val audioTrack = AudioTrack.Builder()
    333 -            .setAudioAttributes(AudioAttributes.Builder().setUsage(Audi
         oAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SP
         EECH).build())
    334 -            .setAudioFormat(AudioFormat.Builder().setSampleRate(sampleR
         ate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setEncoding(AudioForm
         at.ENCODING_PCM_FLOAT).build())
    335 -            .setBufferSizeInBytes(bufferSize)
    336 -            .setTransferMode(AudioTrack.MODE_STREAM)
    337 -            .build()
    338 -
    339 -        audioTrack.play()
    340 -        audioTrack.write(samples, 0, samples.size, AudioTrack.WRITE_BLO
         CKING)
    341 -        audioTrack.stop()
    342 -        audioTrack.release()
    343 -    }
    344 -
    345 -    private fun logAndBroadcastMetrics() {
    346 -        val avgLatency = if (totalFrames > 0) vadLatencySum / totalFram
         es else 0
    347 -        val speechRate = if (totalFrames > 0) speechFrames.toFloat() /
         totalFrames else 0f
    348 -        val metrics = """
    349 -            VAD Performance:
    350 -            Frames: $totalFrames/$speechFrames
    351 -            Speech Rate: ${"%.1f".format(speechRate * 100)}%
    352 -            Avg Latency: ${avgLatency}ms
    353 -            Mode: ${when (vadMode) {
    354 -                0 -> "Ultra Patient"
    355 -                1 -> "Patient"
    356 -                2 -> "Normal"
    357 -                3 -> "Aggressive"
    358 -                else -> "Custom"
    359 -            }}
    360 -        """.trimIndent()
    361 -        val intent = Intent("VAD_METRICS_UPDATE").putExtra("metrics", m
         etrics)
    362 -        localBroadcastManager.sendBroadcast(intent)
    363 -    }
    364 -
    365 -    private fun broadcastStatus(status: String) {
    366 -        val intent = Intent("SYSTEM_STATUS_UPDATE").putExtra("status",
         status)
    367 -        localBroadcastManager.sendBroadcast(intent)
    368 -    }
    369 -
    370 -    fun cleanup() {
    371 -        try {
    372 -            nsWrapper?.nativeFree()
    373 -        } catch (_: Exception) { }
    374 -        try {
    375 -            sileroVad.close()
    376 -        } catch (_: Exception) { }
    377 -    }
    378 -}

