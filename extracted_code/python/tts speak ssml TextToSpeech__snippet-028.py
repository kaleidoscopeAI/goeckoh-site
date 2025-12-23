package com.exocortex.neuroacoustic

import android.Manifest
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.audiofx.NoiseSuppressor
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import be.tarsos.dsp.AudioDispatcher
import be.tarsos.dsp.AudioEvent
import be.tarsos.dsp.AudioProcessor
import be.tarsos.dsp.io.jvm.AudioDispatcherFactory
import be.tarsos.dsp.pitch.PitchDetectionHandler
import be.tarsos.dsp.pitch.PitchDetectionResult
import be.tarsos.dsp.pitch.PitchProcessor
import be.tarsos.dsp.pitch.PitchShifter
import be.tarsos.dsp.gain.GainProcessor
import com.github.pemistahl.lingua.api.Language
import com.github.pemistahl.lingua.api.LanguageDetectorBuilder
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceOptions
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations
import org.apache.commons.math3.ode.nonstiff.DormandPrince853Integrator
import org.silero.vad.SileroVad
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ShortBuffer
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

class MainActivity : AppCompatActivity() {
    private lateinit var statusText: TextView
    private lateinit var startButton: Button
    private lateinit var vadModeSpinner: Spinner
    private lateinit var silenceDurationSeekBar: SeekBar
    private lateinit var vadThresholdSeekBar: SeekBar
    private lateinit var applyVadButton: Button
    private lateinit var sharedPrefs: SharedPreferences
    private val RECORD_REQUEST_CODE = 101
    private var isServiceRunning = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.id_status)
        startButton = findViewById(R.id.id_start)
        vadModeSpinner = findViewById(R.id.vad_mode_spinner)
        silenceDurationSeekBar = findViewById(R.id.silence_duration_seekbar)
        vadThresholdSeekBar = findViewById(R.id.vad_threshold_seekbar)
        applyVadButton = findViewById(R.id.apply_vad_button)

        sharedPrefs = getSharedPreferences("vad_prefs", Context.MODE_PRIVATE)

        // Setup UI
        val modes = arrayOf("NORMAL", "LOW_BITRATE", "AGGRESSIVE", "VERY_AGGRESSIVE")
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, modes)
        vadModeSpinner.adapter = adapter
        vadModeSpinner.setSelection(sharedPrefs.getInt("vad_mode", 0))

        silenceDurationSeekBar.progress = sharedPrefs.getInt("silence_ms", 1200) / 100 // 0-30 for 0-3000ms
        silenceDurationSeekBar.max = 30

        vadThresholdSeekBar.progress = (sharedPrefs.getFloat("vad_threshold", 0.5f) * 100).toInt() // 0-100 for 0.0-1.0
        vadThresholdSeekBar.max = 100

        applyVadButton.setOnClickListener {
            val modeIndex = vadModeSpinner.selectedItemPosition
            val silenceMs = silenceDurationSeekBar.progress * 100
            val threshold = vadThresholdSeekBar.progress / 100f
            sharedPrefs.edit().putInt("vad_mode", modeIndex).putInt("silence_ms", silenceMs).putFloat("vad_threshold", threshold).apply()
            Toast.makeText(this, "VAD settings applied", Toast.LENGTH_SHORT).show()
        }

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
        // (unchanged)
    }

    override fun onDestroy() {
        super.onDestroy()
    }
