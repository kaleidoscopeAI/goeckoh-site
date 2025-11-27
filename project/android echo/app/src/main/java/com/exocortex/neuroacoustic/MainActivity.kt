package com.exocortex.neuroacoustic

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager

class MainActivity : AppCompatActivity() {
    private lateinit var statusText: TextView
    private lateinit var startButton: Button
    private lateinit var vadModeSpinner: Spinner
    private lateinit var silenceDurationSeekBar: SeekBar
    private lateinit var vadThresholdSeekBar: SeekBar
    private lateinit var applyVadButton: Button
    private lateinit var vadMetricsText: TextView
    private lateinit var silenceValueText: TextView
    private lateinit var thresholdValueText: TextView
    private lateinit var sharedPrefs: SharedPreferences
    private var isServiceRunning = false
    private val recordRequestCode = 101

    private val metricsReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val metrics = intent?.getStringExtra("metrics") ?: return
            vadMetricsText.text = metrics
        }
    }

    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val status = intent?.getStringExtra("status") ?: return
            statusText.text = status
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.id_status)
        startButton = findViewById(R.id.id_start)
        vadModeSpinner = findViewById(R.id.vad_mode_spinner)
        silenceDurationSeekBar = findViewById(R.id.silence_duration_seekbar)
        vadThresholdSeekBar = findViewById(R.id.vad_threshold_seekbar)
        applyVadButton = findViewById(R.id.apply_vad_button)
        vadMetricsText = findViewById(R.id.vad_metrics_text)
        silenceValueText = findViewById(R.id.silence_value)
        thresholdValueText = findViewById(R.id.threshold_value)

        sharedPrefs = getSharedPreferences("vad_prefs", Context.MODE_PRIVATE)
        setupVADControls()

        LocalBroadcastManager.getInstance(this).apply {
            registerReceiver(metricsReceiver, IntentFilter("VAD_METRICS_UPDATE"))
            registerReceiver(statusReceiver, IntentFilter("SYSTEM_STATUS_UPDATE"))
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), recordRequestCode)
        } else {
            initSystem()
        }

        startButton.setOnClickListener {
            if (!isServiceRunning) {
                startService(Intent(this, ExocortexService::class.java))
                startButton.text = "Stop Listening"
            } else {
                stopService(Intent(this, ExocortexService::class.java))
                startButton.text = "Start Listening"
            }
            isServiceRunning = !isServiceRunning
        }
    }

    private fun setupVADControls() {
        val modes = arrayOf("Ultra Patient", "Patient", "Normal", "Aggressive", "Ultra Aggressive")
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, modes)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        vadModeSpinner.adapter = adapter
        vadModeSpinner.setSelection(sharedPrefs.getInt("vad_mode", 2))

        silenceDurationSeekBar.max = 30
        silenceDurationSeekBar.progress = sharedPrefs.getInt("silence_ms", 1200) / 100
        silenceValueText.text = "${silenceDurationSeekBar.progress * 100} ms"

        vadThresholdSeekBar.max = 100
        vadThresholdSeekBar.progress = (sharedPrefs.getFloat("vad_threshold", 0.5f) * 100).toInt()
        thresholdValueText.text = String.format("%.2f", vadThresholdSeekBar.progress / 100f)

        silenceDurationSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                silenceValueText.text = "${progress * 100} ms"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        vadThresholdSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                thresholdValueText.text = String.format("%.2f", progress / 100f)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        applyVadButton.setOnClickListener {
            val modeIndex = vadModeSpinner.selectedItemPosition
            val silenceMs = silenceDurationSeekBar.progress * 100
            val threshold = vadThresholdSeekBar.progress / 100f
            sharedPrefs.edit()
                .putInt("vad_mode", modeIndex)
                .putInt("silence_ms", silenceMs)
                .putFloat("vad_threshold", threshold)
                .apply()
            Toast.makeText(this, "VAD settings applied", Toast.LENGTH_SHORT).show()
            sendBroadcast(Intent("VAD_SETTINGS_APPLIED"))
        }
    }

    private fun initSystem() {
        statusText.text = "Preparing models..."
        try {
            AssetInstaller.installAll(applicationContext, statusText)
            statusText.text = "System Initialized"
        } catch (e: Exception) {
            statusText.text = "Init failed: ${e.message}"
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == recordRequestCode && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            initSystem()
        } else {
            Toast.makeText(this, "Audio permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        LocalBroadcastManager.getInstance(this).apply {
            unregisterReceiver(metricsReceiver)
            unregisterReceiver(statusReceiver)
        }
        super.onDestroy()
    }
}
