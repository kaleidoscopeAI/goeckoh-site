private lateinit var statusText: TextView
private lateinit var startButton: Button
private lateinit var tts: TextToSpeech
private val RECORD_REQUEST_CODE = 101
private var isServiceRunning = false

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    statusText = findViewById(R.id.id_status)
    startButton = findViewById(R.id.id_start)
    tts = TextToSpeech(this, this)

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

override fun onInit(status: Int) {
    if (status == TextToSpeech.SUCCESS) {
        tts.language = Locale.US
    } else {
        Toast.makeText(this, "TTS Initialization Failed", Toast.LENGTH_SHORT).show()
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
    // Check for LLM model
    val modelFile = File(filesDir, "llm/model.task")
    if (!modelFile.exists()) {
        statusText.text = "LLM model missing. Download Gemma-3 1B and place in app files."
        // In production, add download logic or instructions
    } else {
        statusText.text = "System Initialized"
    }
}

override fun onDestroy() {
    tts.shutdown()
    super.onDestroy()
}
