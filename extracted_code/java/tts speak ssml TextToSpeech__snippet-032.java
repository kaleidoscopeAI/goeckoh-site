private lateinit var neuroAcousticMirror: NeuroAcousticMirror
private lateinit var crystallineHeart: CrystallineHeart
private lateinit var gatedAGI: GatedAGI
private lateinit var statusText: TextView
private lateinit var startButton: Button
private lateinit var tts: TextToSpeech
private val RECORD_REQUEST_CODE = 101

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main) // Assume a layout with TextView id_status and Button id_start

    statusText = findViewById(R.id.id_status)
    startButton = findViewById(R.id.id_start)
    tts = TextToSpeech(this, this)

    // Check permissions
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), RECORD_REQUEST_CODE)
    } else {
        initSystem()
    }

    startButton.setOnClickListener {
        startListening()
    }
}

override fun onInit(status: Int) {
    if (status == TextToSpeech.SUCCESS) {
        tts.language = Locale.US
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
    neuroAcousticMirror = NeuroAcousticMirror(this, tts)
    crystallineHeart = CrystallineHeart(1024)
    gatedAGI = GatedAGI(crystallineHeart)
    statusText.text = "System Initialized"
}

private fun startListening() {
    Executors.newSingleThreadExecutor().execute {
        while (true) {
            neuroAcousticMirror.listenAndProcess { correctedText, prosody ->
                runOnUiThread { statusText.text = "GCL: ${crystallineHeart.gcl}" }
                val gcl = crystallineHeart.updateAndGetGCL(prosody.arousal, prosody.volume)
                gatedAGI.executeBasedOnGCL(gcl, correctedText)
            }
        }
    }
}

override fun onDestroy() {
    tts.shutdown()
    super.onDestroy()
}
