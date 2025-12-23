  private lateinit var whisper: Whisper
  private lateinit var recorder: Recorder
  private lateinit var sileroVad: SileroVad
  private lateinit var sharedPrefs: SharedPreferences
  private var partialText = ""
  private var silenceStart: Long = 0
  private var isSpeechDetected = false
  private var vadThreshold: Float = 0.5f
  private var silenceDurationMs: Int = 1200
  private var vadMode: Int = 1 // Not used in Silero, but for adaptive
  private var noiseSuppressor: NoiseSuppressor? = null
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

      // Init Noise Suppressor
      if (NoiseSuppressor.isAvailable()) {
          noiseSuppressor =
