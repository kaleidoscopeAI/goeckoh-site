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

      if (ContextCompat.checkSelfPermission(this,
