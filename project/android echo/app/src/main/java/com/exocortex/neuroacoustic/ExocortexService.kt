package com.exocortex.neuroacoustic

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import java.util.concurrent.Executors

class ExocortexService : Service() {
    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var mirror: NeuroAcousticMirror
    private lateinit var heart: CrystallineHeart
    private lateinit var agi: GatedAGI

    override fun onCreate() {
        super.onCreate()
        mirror = NeuroAcousticMirror(this)
        heart = CrystallineHeart(1024)
        agi = GatedAGI(this, heart, mirror)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        executor.execute {
            while (!executor.isShutdown) {
                try {
                    mirror.listenAndProcess { correctedText, prosody ->
                        val gcl = heart.updateAndGetGCL(prosody.arousal, prosody.volume, prosody.pitchVariance)
                        agi.executeBasedOnGCL(gcl, correctedText)
                    }
                } catch (e: Exception) {
                    Log.e("ExocortexService", "Loop error: ${e.message}")
                }
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        executor.shutdownNow()
        mirror.cleanup()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
