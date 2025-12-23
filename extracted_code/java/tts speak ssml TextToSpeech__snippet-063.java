  private lateinit var neuroAcousticMirror: NeuroAcousticMirror
  private lateinit var crystallineHeart: CrystallineHeart
  private lateinit var gatedAGI: GatedAGI
  private lateinit var statusText: TextView
  private lateinit var startButton: Button
  private lateinit var tts: TextToSpeech
  private val RECORD_REQUEST_CODE = 101

  override fun onCreate(savedInstanceState: Bundle?) {
      super.onCreate(savedInstanceState)
      setContentView(R.layout.activity_main) // Assume a layout with
