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

    val sttModels = listOf(
        "vosk-model-small-en-us-0.15.zip" to "en",
        "vosk-model-small-es-0.42.zip" to "es",
        "vosk-model-small-fr-0.22.zip" to "fr"
    )
    sttModels.forEach { (zipName, lang) ->
        val modelDir = File(filesDir, "vosk-model-small-$lang")
        if (!modelDir.exists()) {
            try {
                assets.open(zipName).use { input ->
                    ZipInputStream(input).use { zip ->
                        var entry = zip.nextEntry
                        while (entry != null) {
                            val file = File(modelDir, entry.name)
                            if (entry.isDirectory) {
                                file.mkdirs()
                            } else {
                                file.parentFile?.mkdirs()
                                FileOutputStream(file).use { output ->
                                    zip.copyTo(output)
                                }
                            }
                            entry = zip.nextEntry
                        }
                    }
                }
            } catch (e: Exception) {
                statusText.text = "STT model unzip failed for $lang: ${e.message}"
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
    super.onDestroy()
}
