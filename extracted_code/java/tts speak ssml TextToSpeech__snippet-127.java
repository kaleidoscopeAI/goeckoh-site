  private lateinit var whisper: Whisper
  private lateinit var recorder: Recorder
  private lateinit var sileroVad: SileroVad
  private lateinit var sharedPrefs: SharedPreferences
  private var partialText = ""
  private var silenceStart: Long = 0
  private var isSpeechDetected = false
  private var vadThreshold: Float = 0.5f
  private var silenceDurationMs: Int = 1200
  private var vadMode: Int = 1
  private var nsWrapper: NsWrapper? = null
  private val localBroadcastManager =
