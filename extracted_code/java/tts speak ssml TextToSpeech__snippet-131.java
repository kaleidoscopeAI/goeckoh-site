42      private val languageDetector = LanguageDetectorBuilder.fromLanguages
    (Language.ENGLISH, Language.SPANISH, Language.FRENCH).build()
43 -    private val whisper = Whisper(context)
44 -    private val recorder = Recorder(context)
45 -    private val sileroVad: SileroVad = SileroVad.load(context)
43      private val sharedPrefs: SharedPreferences = context.getSharedPrefer
    ences("vad_prefs", Context.MODE_PRIVATE)
   ⋮
45
49 -    private var partialText = ""
50 -    private var silenceStart: Long = 0
51 -    private var isSpeechDetected = false
46 +    private val models = mutableMapOf<String, Model>()
47 +    private var vad = VadWebRTC.builder()
48 +        .setSampleRate(SampleRate.SAMPLE_RATE_16K)
49 +        .setFrameSize(FrameSize.FRAME_SIZE_320)
50 +        .setMode(Mode.NORMAL)
51 +        .setSilenceDurationMs(1200)
52 +        .setSpeechDurationMs(100)
53 +        .build()
54 +
55 +    private var vadModeIndex = 2
56 +    private var silenceDurationMs: Int = 1200
57      private var vadThreshold: Float = 0.5f
53 -    private var silenceDurationMs: Int = 1200
54 -    private var vadMode: Int = 2
55 -    private var nsWrapper: NsWrapper? = null
56 -    private var useNoiseSuppression = true
58 +    private var noiseSuppressor: NoiseSuppressor? = null
59 +
60      private var totalFrames = 0
   ⋮
62      private var vadLatencySum = 0L
60 -    private var lastMetricsUpdate = 0L
63

