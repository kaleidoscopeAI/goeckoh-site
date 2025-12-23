package com.kaleidoscope.body
import android.util.Log
class ControlMapper(private val hal: DeviceHAL) {
private val TAG = "ControlMapper"
private val mapper = NeuralHardwareMapper(maxFreq = 2000f)
