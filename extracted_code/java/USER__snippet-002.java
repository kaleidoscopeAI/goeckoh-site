private lateinit var neuroAcousticMirror: NeuroAcousticMirror
private lateinit var crystallineHeart: CrystallineHeart
private lateinit var gatedAGI: GatedAGI
private val executor = Executors.newSingleThreadExecutor()
private val isRunning = AtomicBoolean(false)

override fun onCreate() {
    super.onCreate()
    neuroAcousticMirror = NeuroAcousticMirror(this)
    crystallineHeart = CrystallineHeart(1024)
    gatedAGI = GatedAGI(this, crystallineHeart, neuroAcousticMirror)
}

override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    isRunning.set(true)
    executor.execute {
        while (isRunning.get()) {
            try {
                neuroAcousticMirror.listenAndProcess { correctedText, prosody ->
                    val gcl = crystallineHeart.updateAndGetGCL(prosody.arousal, prosody.volume, prosody.pitchVariance)
                    gatedAGI.executeBasedOnGCL(gcl, correctedText)
                }
            } catch (e: Exception) {
                Log.e("Service", "Error in processing loop: ${e.message}")
                // Brief pause before retry
                Thread.sleep(1000)
            }
        }
    }
    return START_STICKY
}

override fun onBind(intent: Intent?): IBinder? = null

override fun onDestroy() {
    isRunning.set(false)
    executor.shutdownNow()
    neuroAcousticMirror.cleanup()
    super.onDestroy()
}
