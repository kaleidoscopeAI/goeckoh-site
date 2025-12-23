private lateinit var statusText: TextView
private lateinit var startButton: Button
private lateinit var vadModeSpinner: Spinner
private lateinit var silenceDurationSeekBar: SeekBar
private lateinit var vadThresholdSeekBar: SeekBar
private lateinit var applyVadButton: Button
private lateinit var vadMetricsText: TextView
private lateinit var noiseSuppressionToggle: Switch
private lateinit var sharedPrefs: SharedPreferences
private val RECORD_REQUEST_CODE = 101
private var isServiceRunning = false
private val localBroadcastManager by lazy { LocalBroadcastManager.getInstance(this) }

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    initializeViews()
    setupUI()
    setupBroadcastReceiver()

    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), RECORD_REQUEST_CODE)
    } else {
        initSystem()
    }
}

private fun initializeViews() {
    statusText = findViewById(R.id.id_status)
    startButton = findViewById(R.id.id_start)
    vadModeSpinner = findViewById(R.id.vad_mode_spinner)
    silenceDurationSeekBar = findViewById(R.id.silence_duration_seekbar)
    vadThresholdSeekBar = findViewById(R.id.vad_threshold_seekbar)
    applyVadButton = findViewById(R.id.apply_vad_button)
    vadMetricsText = findViewById(R.id.vad_metrics_text)
    noiseSuppressionToggle = findViewById(R.id.noise_suppression_toggle)

    sharedPrefs = getSharedPreferences("vad_prefs", Context.MODE_PRIVATE)
}

private fun setupUI() {
    val modes = arrayOf("ULTRA_PATIENT", "PATIENT", "NORMAL", "AGGRESSIVE", "ULTRA_AGGRESSIVE")
    val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, modes)
    vadModeSpinner.adapter = adapter
    vadModeSpinner.setSelection(sharedPrefs.getInt("vad_mode", 2))

    silenceDurationSeekBar.progress = sharedPrefs.getInt("silence_ms", 1200) / 100
    silenceDurationSeekBar.max = 50 // 0-5000ms

    vadThresholdSeekBar.progress = (sharedPrefs.getFloat("vad_threshold", 0.5f) * 100).toInt()
    vadThresholdSeekBar.max = 100

    noiseSuppressionToggle.isChecked = sharedPrefs.getBoolean("noise_suppression", true)

    applyVadButton.setOnClickListener {
        val modeIndex = vadModeSpinner.selectedItemPosition
        val silenceMs = silenceDurationSeekBar.progress * 100
        val threshold = vadThresholdSeekBar.progress / 100f
        val noiseSuppression = noiseSuppressionToggle.isChecked

        sharedPrefs.edit()
            .putInt("vad_mode", modeIndex)
            .putInt("silence_ms", silenceMs)
            .putFloat("vad_threshold", threshold)
            .putBoolean("noise_suppression", noiseSuppression)
            .apply()

        Toast.makeText(this, "VAD settings applied", Toast.LENGTH_SHORT).show()
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

private fun setupBroadcastReceiver() {
    localBroadcastManager.registerReceiver(metricsReceiver, IntentFilter("VAD_METRICS_UPDATE"))
}

private val metricsReceiver = object : BroadcastReceiver() {
    override fun onReceive(context: Context?, intent: Intent?) {
        when (intent?.action) {
            "VAD_METRICS_UPDATE" -> {
                val metrics = intent.getStringExtra("metrics") ?: ""
                runOnUiThread { vadMetricsText.text = metrics }
            }
            "SYSTEM_STATUS_UPDATE" -> {
                val status = intent.getStringExtra("status") ?: ""
                runOnUiThread { statusText.text = status }
            }
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
    // Initialize TTS models
    val ttsModels = listOf(
        "vits-piper-en_US-amy-medium.onnx" to "en",
        "vits-piper-es_ES-mls_9972-medium.onnx" to "es",
        "vits-piper-fr_FR-upmc-medium.onnx" to "fr"
    )

    ttsModels.forEach { (modelName, lang) ->
        val ttsModelFile = File(filesDir, modelName)
        if (!ttsModelFile.exists()) {
            try {
                assets.open(modelName).use { input ->
                    FileOutputStream(ttsModelFile).use { output ->
                        input.copyTo(output)
                    }
                }
            } catch (e: Exception) {
                statusText.text = "TTS model copy failed for $lang: ${e.message}"
                return
            }
        }
    }

    // Initialize Whisper models
    val whisperAssets = listOf("whisper-tiny.tflite", "filters_vocab_multilingual.bin")
    whisperAssets.forEach { fileName ->
        val file = File(filesDir, fileName)
        if (!file.exists()) {
            try {
                assets.open(fileName).use { input ->
                    FileOutputStream(file).use { output ->
                        input.copyTo(output)
                    }
                }
            } catch (e: Exception) {
                statusText.text = "Whisper asset copy failed: ${e.message}"
                return
            }
        }
    }

    val llmModelFile = File(filesDir, "gemma-1.1-2b-it-q4f16.task")
    if (!llmModelFile.exists()) {
        statusText.text = "LLM model missing. Download Gemma and place in app files."
        return
    }

    statusText.text = "System Initialized"
}

override fun onDestroy() {
    localBroadcastManager.unregisterReceiver(metricsReceiver)
    super.onDestroy()
}
