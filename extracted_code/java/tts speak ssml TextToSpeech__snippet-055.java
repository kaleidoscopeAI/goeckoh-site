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
private val localBroadcastManager = LocalBroadcastManager.getInstance(this)

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    statusText = findViewById(R.id.id_status)
    startButton = findViewById(R.id.id_start)
    vadModeSpinner = findViewById(R.id.vad_mode_spinner)
    silenceDurationSeekBar = findViewById(R.id.silence_duration_seekbar)
    vadThresholdSeekBar = findViewById(R.id.vad_threshold_seekbar)
    applyVadButton = findViewById(R.id.apply_vad_button)
    vadMetricsText = findViewById(R.id.vad_metrics_text)

    sharedPrefs = getSharedPreferences("vad_prefs", Context.MODE_PRIVATE)

    // Setup UI (unchanged)
    // ...

    applyVadButton.setOnClickListener {
        // (unchanged)
    }

    // Register broadcast receiver for metrics
    localBroadcastManager.registerReceiver(metricsReceiver, IntentFilter("VAD_METRICS_UPDATE"))

    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), RECORD_REQUEST_CODE)
    } else {
        initSystem()
    }

    startButton.setOnClickListener {
        // (unchanged)
    }
}

private val metricsReceiver = object : BroadcastReceiver() {
    override fun onReceive(context: Context?, intent: Intent?) {
        val metrics = intent?.getStringExtra("metrics") ?: ""
        vadMetricsText.text = metrics
    }
}

override fun onDestroy() {
    localBroadcastManager.unregisterReceiver(metricsReceiver)
    super.onDestroy()
}

private fun initSystem() {
    // (unchanged)
}
