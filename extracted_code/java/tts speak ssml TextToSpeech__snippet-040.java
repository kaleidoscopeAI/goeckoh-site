private lateinit var statusText: TextView
private lateinit var startButton: Button
private val RECORD_REQUEST_CODE = 101
private var isServiceRunning = false

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    statusText = findViewById(R.id.id_status)
    startButton = findViewById(R.id.id_start)

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
    // Check models
    val ttsModelFile = File(filesDir, "vits-piper-en_US-amy-medium.onnx")
    if (!ttsModelFile.exists()) {
        try {
            assets.open("vits-piper-en_US-amy-medium.onnx").use { input ->
                FileOutputStream(ttsModelFile).use { output ->
                    input.copyTo(output)
                }
            }
        } catch (e: Exception) {
            statusText.text = "TTS model copy failed: ${e.message}"
            return
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
    super.onDestroy()
}
