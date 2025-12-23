package com.exocortex.neuroacoustic

import android.Manifest
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import be.tarsos.dsp.AudioDispatcher
import be.tarsos.dsp.AudioEvent
import be.tarsos.dsp.AudioProcessor
import be.tarsos.dsp.pitch.PitchDetectionHandler
import be.tarsos.dsp.pitch.PitchDetectionResult
import be.tarsos.dsp.pitch.PitchProcessor
import be.tarsos.dsp.io.jvm.AudioDispatcherFactory
import be.tarsos.dsp.pitch.PitchShift
import be.tarsos.dsp.resample.RateTransposer
import be.tarsos.dsp.gain.GainProcessor
import com.k2fsa.sherpa.onnx.TtsConfig
import com.k2fsa.sherpa.onnx.OfflineTts
import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations
import org.apache.commons.math3.ode.nonstiff.DormandPrince853Integrator
import org.kaldi.Model
import org.kaldi.RecognitionListener
import org.kaldi.Vosk
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceOptions
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

class MainActivity : AppCompatActivity() {
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
