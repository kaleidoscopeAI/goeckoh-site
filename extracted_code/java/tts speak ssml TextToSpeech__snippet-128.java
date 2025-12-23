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

      // Init RNNoise via JNI
      System.loadLibrary("webrtc_ns_jni")
      nsWrapper = NsWrapper()
      nsWrapper!!.nativeCreate()

      // Load prefs
      sharedPrefs = context.getSharedPreferences("vad_prefs",
