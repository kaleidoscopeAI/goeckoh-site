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
    val modes = arrayOf("NORMAL", "LOW_BITRATE", "AGGRESSIVE", "VERY_AGGRESSIVE")
    val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, modes)
    vadModeSpinner.adapter = adapter
    vadModeSpinner.setSelection(sharedPrefs.getInt("vad_mode", 0))

    silenceDurationSeekBar.progress = sharedPrefs.getInt("silence_ms", 1200) / 100 // 0-30 for 0-3000ms
    silenceDurationSeekBar.max = 30

    vadThresholdSeekBar.progress = (sharedPrefs.getFloat("vad_threshold", 0.5f) * 100).toInt() // 0-100 for 0.0-1.0
    vadThresholdSeekBar.max = 100

    applyVadButton.setOnClickListener {
        val modeIndex = vadModeSpinner.selectedItemPosition
        val silenceMs = silenceDurationSeekBar.progress * 100
        val threshold = vadThresholdSeekBar.progress / 100f
        sharedPrefs.edit().putInt("vad_mode", modeIndex).putInt("silence_ms", silenceMs).putFloat("vad_threshold", threshold).apply()
        Toast.makeText(this, "VAD settings applied", Toast.LENGTH_SHORT).show()
    }

    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), RECORD_REQUEST_CODE)
    } else {
        initSystem()
    }

    startButton.setOnClickListener {
        if (!isServiceRunning) {
            startService(Intent(this, ExocortexService::class.java))
            startButton.text = "Stop Listening"
            isServiceRunning = true
        } else {
            stopService(Intent(this, ExocortexService::class.java))
            startButton.text = "Start Listening"
            isServiceRunning = false
        }
    }
}

override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (requestCode == RECORD_REQUEST_CODE && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        initSystem()
    } else {
        Toast.makeText(this, "Audio permission denied", Toast.LENGTH_SHORT).show()
    }
}

private fun initSystem() {
    // (unchanged)
}

override fun onDestroy() {
    super.onDestroy()
}
