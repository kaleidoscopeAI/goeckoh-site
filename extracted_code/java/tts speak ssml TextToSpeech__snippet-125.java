  private lateinit var statusText: TextView
  private lateinit var startButton: Button
  private lateinit var vadModeSpinner: Spinner
  private lateinit var silenceDurationSeekBar: SeekBar
  private lateinit var vadThresholdSeekBar: SeekBar
  private lateinit var applyVadButton: Button
  private lateinit var vadMetricsText: TextView
  private lateinit var sharedPrefs: SharedPreferences
  private val RECORD_REQUEST_CODE = 101
  private var isServiceRunning = false
  private val localBroadcastManager =
