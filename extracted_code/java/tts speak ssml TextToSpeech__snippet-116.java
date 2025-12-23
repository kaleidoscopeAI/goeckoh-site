  private lateinit var statusText: TextView
  private lateinit var startButton: Button
  private lateinit var vadModeSpinner: Spinner
  private lateinit var silenceDurationSeekBar: SeekBar
  private lateinit var vadThresholdSeekBar: SeekBar
  private lateinit var applyVadButton: Button
  private lateinit var sharedPrefs: SharedPreferences
  private val RECORD_REQUEST_CODE = 101
  private var isServiceRunning = false

  override fun onCreate(savedInstanceState: Bundle?) {
      super.onCreate(savedInstanceState)
      setContentView(R.layout.activity_main)

      statusText = findViewById(R.id.id_status)
      startButton = findViewById(R.id.id_start)
      vadModeSpinner = findViewById(R.id.vad_mode_spinner)
      silenceDurationSeekBar = findViewById(R.id.silence_duration_seekbar)
      vadThresholdSeekBar = findViewById(R.id.vad_threshold_seekbar)
      applyVadButton = findViewById(R.id.apply_vad_button)

      sharedPrefs = getSharedPreferences("vad_prefs", Context.MODE_PRIVATE)

      // Setup UI
      val modes = arrayOf("NORMAL", "LOW_BITRATE", "AGGRESSIVE",
