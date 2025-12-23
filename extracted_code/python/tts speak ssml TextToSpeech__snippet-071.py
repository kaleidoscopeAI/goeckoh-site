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
 44 +    private val languageDetector = LanguageDetectorBuilder.fromLanguages(Language.ENGLISH, Language.SPANISH, Language.F
     RENCH).build()
 45 +    private val sharedPrefs: SharedPreferences = context.getSharedPreferences("vad_prefs", Context.MODE_PRIVATE)
 46 +    private val localBroadcastManager = LocalBroadcastManager.getInstance(context)
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
 80 +            val config = OfflineTtsConfig(model = path, numThreads = 1, debug = false)
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
 93 +            val base = File(context.filesDir, dir)
 94 +            val modelDir = resolveModelDir(base)
 95 +            if (modelDir != null && modelDir.exists()) {
 96 +                models[lang] = Model(modelDir.absolutePath)
 97 +            } else {
 98 +                Log.w("Mirror", "Vosk model missing for $lang at ${base.absolutePath}")
 99 +            }
100 +        }
101 +    }
102 +
103 +    private fun resolveModelDir(base: File): File? {
104 +        if (!base.exists()) return null
105 +        val children = base.listFiles()
106 +        return if (children != null && children.size == 1 && children[0].isDirectory) children[0] else base
107 +    }
108 +
109 +    private fun updateVadFromPrefs() {
110 +        vadModeIndex = sharedPrefs.getInt("vad_mode", 2)
111 +        silenceDurationMs = sharedPrefs.getInt("silence_ms", 1200)
112 +        vadThreshold = sharedPrefs.getFloat("vad_threshold", 0.5f)
113 +        rebuildVad()
114 +    }
115 +
116 +    private fun rebuildVad() {
117 +        vad.close()
118 +        val mode = when (vadModeIndex) {
119 +            0 -> Mode.LOW_BITRATE
120 +            1 -> Mode.NORMAL
121 +            3 -> Mode.AGGRESSIVE
122 +            4 -> Mode.VERY_AGGRESSIVE
123 +            else -> Mode.NORMAL
124 +        }
125 +        vad = VadWebRTC.builder()
126 +            .setSampleRate(SampleRate.SAMPLE_RATE_16K)
127 +            .setFrameSize(FrameSize.FRAME_SIZE_320)
128 +            .setMode(mode)
129 +            .setSilenceDurationMs(silenceDurationMs)
130 +            .setSpeechDurationMs(100)
131 +            .build()
132 +    }
133 +
134 +    fun tuneVAD(mode: Int, silenceMs: Int, threshold: Float) {
135 +        vadModeIndex = mode
136 +        silenceDurationMs = silenceMs
137 +        vadThreshold = threshold.coerceIn(0.1f, 0.9f)
138 +        rebuildVad()
139 +    }
140 +
141 +    fun listenAndProcess(callback: (String, Prosody) -> Unit) {
142 +        if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.RECORD_AUDIO) != android.content.pm.
     PackageManager.PERMISSION_GRANTED) {
143 +            Log.w("Mirror", "Audio permission not granted.")
144 +            return
145 +        }
146 +        if (models.isEmpty()) {
147 +            Log.w("Mirror", "No Vosk models loaded.")
148 +            return
149 +        }
150 +        resetState()
151 +
152 +        val recognizers: Map<String, Recognizer> = models.mapValues { (_, model) ->
153 +            org.kaldi.Vosk.createRecognizer(model, sampleRate.toFloat())
154 +        }
155 +
156 +        val audioRecord = AudioRecord(
157 +            MediaRecorder.AudioSource.MIC,
158 +            sampleRate,
159 +            AudioFormat.CHANNEL_IN_MONO,
160 +            AudioFormat.ENCODING_PCM_16BIT,
161 +            AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
162 +        )
163 +
164 +        if (NoiseSuppressor.isAvailable()) {
165 +            noiseSuppressor = NoiseSuppressor.create(audioRecord.audioSessionId)
166 +        }
167 +
168 +        audioRecord.startRecording()
169 +        broadcastStatus("Listening...")
170 +        val buffer = ShortArray(512)
171 +
172 +        while (true) {
173 +            val read = audioRecord.read(buffer, 0, buffer.size)
174 +            if (read <= 0) continue
175 +
176 +            val frame = buffer.copyOf(read)
177 +            totalFrames++
178 +            val frameBytes = ShortArrayToByteArray(frame)
179 +            val rms = calculateRms(frame)
180 +            val isSpeech = vad.isSpeech(frameBytes) || rms > (vadThreshold * 0.1)
181 +
182 +            if (isSpeech) {
183 +                speechFrames++
184 +                silenceStart = 0
185 +            } else if (silenceStart == 0L && speechFrames > 0) {
186 +                silenceStart = System.currentTimeMillis()
187 +            }
188 +
189 +            recognizers.values.forEach { it.acceptWaveForm(frame, frame.size) }
190 +            collectForProsody(frame)
191 +
192 +            if (silenceStart > 0 && System.currentTimeMillis() - silenceStart >= silenceDurationMs) {
193 +                audioRecord.stop()
194 +                audioRecord.release()
195 +                noiseSuppressor?.release()
196 +
197 +                val best = selectBestResult(recognizers)
198 +                if (best.first.isNotEmpty()) {
199 +                    val rawText = best.first
200 +                    val prosody = extractProsody(audioBuffer.toByteArray(), rawText)
201 +                    val correctedText = correctToFirstPerson(rawText)
202 +                    speakCorrectedText(correctedText, prosody)
203 +                    callback(correctedText, prosody)
204 +                    logAndBroadcastMetrics()
205 +                }
206 +                break
207 +            }
208 +        }
209 +    }
210 +
211 +    private fun resetState() {
212 +        audioBuffer.reset()
213 +        totalFrames = 0
214 +        speechFrames = 0
215 +        silenceStart = 0
216 +    }
217 +
218 +    private fun ShortArrayToByteArray(samples: ShortArray): ByteArray {
219 +        val byteArray = ByteArray(samples.size * 2)
220 +        ByteBuffer.wrap(byteArray).asShortBuffer().put(samples)
221 +        return byteArray
222 +    }
223 +
224 +    private fun selectBestResult(recognizers: Map<String, Recognizer>): Pair<String, Double> {
225 +        val results = recognizers.map { (_, rec) ->
226 +            val json = try { rec.finalResult() } catch (_: Exception) { rec.result() }
227 +            val parsed = parseResultWithConfidence(json)
228 +            parsed
229 +        }
230 +        val best = results.maxByOrNull { it.second } ?: return "" to 0.0
231 +        return best
232 +    }
233 +
234 +    private fun parseResultWithConfidence(json: String?): Pair<String, Double> {
235 +        if (json.isNullOrEmpty()) return "" to 0.0
236 +        val obj = JSONObject(json)
237 +        val text = obj.optString("text", "")
238 +        val conf = if (obj.has("result")) {
239 +            val arr = obj.getJSONArray("result")
240 +            (0 until arr.length()).map { arr.getJSONObject(it).optDouble("conf", 0.0) }.average()
241 +        } else 0.0
242 +        return text to if (conf.isNaN()) 0.0 else conf
243 +    }
244 +
245 +    private fun collectForProsody(samples: ShortArray) {
246 +        val byteArray = ByteArray(samples.size * 2)
247 +        ByteBuffer.wrap(byteArray).asShortBuffer().put(samples)
248 +        audioBuffer.write(byteArray)
249 +    }
250 +
251 +    private fun extractProsody(audioBytes: ByteArray, text: String): Prosody {
252 +        if (audioBytes.isEmpty()) return Prosody(120.0, 1.0, 0.5, 0.5, 20.0)
253 +
254 +        val dispatcher = AudioDispatcherFactory.fromByteArray(audioBytes, sampleRate, 1024, 512)
255 +        val pitches = mutableListOf<Float>()
256 +        dispatcher.addAudioProcessor(PitchProcessor(PitchProcessor.PitchEstimationAlgorithm.YIN, sampleRate.toFloat(),
     1024, object : PitchDetectionHandler {
257 +            override fun handlePitch(result: PitchDetectionResult, event: AudioEvent) {
258 +                if (result.pitch > 0) pitches.add(result.pitch)
259 +            }
260 +        }))
261 +        dispatcher.run()
262 +        val avgPitch = if (pitches.isNotEmpty()) pitches.average() else 120.0
263 +        val pitchVariance = if (pitches.size > 1) pitches.map { (it - avgPitch).pow(2) }.average() else 0.0
264 +
265 +        val durationSec = audioBytes.size / (sampleRate * 2).toDouble()
266 +        val wordCount = text.split("\\s+".toRegex()).size.coerceAtLeast(1)
267 +        val rate = if (durationSec > 0) (wordCount / durationSec) / 1.5 else 1.0
268 +
269 +        val shortArray = ShortArray(audioBytes.size / 2)
270 +        ByteBuffer.wrap(audioBytes).asShortBuffer().get(shortArray)
271 +        var sum = 0.0
272 +        shortArray.forEach { sum += (it / 32768.0).pow(2) }
273 +        val volume = if (shortArray.isNotEmpty()) sqrt(sum / shortArray.size) else 0.0
274 +
275 +        val arousal = ((avgPitch / 150.0) + (pitchVariance / 50.0) + (volume * 5.0)) / 3.0
276 +        return Prosody(avgPitch, rate.coerceIn(0.5, 2.0), arousal.coerceIn(0.0, 1.0), volume.coerceIn(0.0, 1.0), pitchV
     ariance)
277 +    }
278 +
279 +    private fun correctToFirstPerson(text: String): String {
280 +        if (text.isEmpty()) return ""
281 +        var corrected = text.replace(Regex("\\b(you|he|she|they)\\b", RegexOption.IGNORE_CASE), "I")
282 +            .replace(Regex("\\b(your|his|her|their)\\b", RegexOption.IGNORE_CASE), "my")
283 +        if (!corrected.matches(Regex("I .*", RegexOption.IGNORE_CASE))) {
284 +            corrected = "I " + corrected.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
285 +        }
286 +        return corrected
287 +    }
288 +
289 +    private fun speakCorrectedText(text: String, prosody: Prosody) {
290 +        if (text.isEmpty()) return
291 +        val detectedLanguage = languageDetector.detectLanguageOf(text) ?: Language.ENGLISH
292 +        val langCode = when (detectedLanguage) {
293 +            Language.SPANISH -> "es"
294 +            Language.FRENCH -> "fr"
295 +            else -> "en"
296 +        }
297 +        val offlineTts = ttsMap[langCode] ?: ttsMap["en"] ?: return
298 +
299 +        val generatedAudio = offlineTts.generate(text, speed = prosody.rate.toFloat(), speakerId = 0)
300 +        val samples = generatedAudio.samples
301 +        val ttsSampleRate = generatedAudio.sampleRate
302 +
303 +        val dispatcher = AudioDispatcherFactory.fromFloatArray(samples, ttsSampleRate, 1024, 512)
304 +        val pitchShiftFactor = (prosody.pitch / 120.0).toFloat()
305 +        val pitchShift = PitchShifter(pitchShiftFactor, ttsSampleRate.toFloat(), 1024, 10)
306 +        val gainProcessor = GainProcessor(prosody.volume * 2.0)
307 +        dispatcher.addAudioProcessor(pitchShift)
308 +        dispatcher.addAudioProcessor(gainProcessor)
309 +
310 +        val processedBuffer = FloatArray(samples.size * 2)
311 +        var index = 0
312 +        dispatcher.addAudioProcessor(object : AudioProcessor {
313 +            override fun process(audioEvent: AudioEvent): Boolean {
314 +                val buffer = audioEvent.floatBuffer
315 +                buffer.copyInto(processedBuffer, index)
316 +                index += buffer.size
317 +                return true
318 +            }
319 +            override fun processingFinished() {}
320 +        })
321 +        dispatcher.run()
322 +        val processedSamples = processedBuffer.copyOf(index)
323 +        playAudio(processedSamples, ttsSampleRate)
324 +    }
325 +
326 +    private fun playAudio(samples: FloatArray, sampleRate: Int) {
327 +        val bufferSize = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM
     _FLOAT)
328 +        val audioTrack = AudioTrack.Builder()
329 +            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAtt
     ributes.CONTENT_TYPE_SPEECH).build())
330 +            .setAudioFormat(AudioFormat.Builder().setSampleRate(sampleRate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO
     ).setEncoding(AudioFormat.ENCODING_PCM_FLOAT).build())
331 +            .setBufferSizeInBytes(bufferSize)
332 +            .setTransferMode(AudioTrack.MODE_STREAM)
333 +            .build()
334 +
335 +        audioTrack.play()
336 +        audioTrack.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
337 +        audioTrack.stop()
338 +        audioTrack.release()
339 +    }
340 +
341 +    private fun logAndBroadcastMetrics() {
342 +        val speechRate = if (totalFrames > 0) speechFrames.toFloat() / totalFrames else 0f
343 +        val metrics = """
344 +            VAD Performance:
345 +            Frames: $totalFrames/$speechFrames
346 +            Speech Rate: ${"%.1f".format(speechRate * 100)}%
347 +            Threshold: ${"%.2f".format(vadThreshold)}
348 +            Silence: ${silenceDurationMs}ms
349 +        """.trimIndent()
350 +        val intent = Intent("VAD_METRICS_UPDATE").putExtra("metrics", metrics)
351 +        localBroadcastManager.sendBroadcast(intent)
352 +    }
353 +
354 +    private fun broadcastStatus(status: String) {
355 +        val intent = Intent("SYSTEM_STATUS_UPDATE").putExtra("status", status)
356 +        localBroadcastManager.sendBroadcast(intent)
357 +    }
358 +
359 +    fun cleanup() {
360 +        try {
361 +            noiseSuppressor?.release()
362 +        } catch (_: Exception) { }
363 +        try {
364 +            vad.close()
365 +        } catch (_: Exception) { }
366 +    }
367 +}

