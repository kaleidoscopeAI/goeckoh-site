package com.kaleidoscope.body
import android.util.Log
import kotlinx.coroutines.*
import kotlin.random.Random
import kotlin.math.max
class KaleidoscopeEngine(val state: SystemState, private val hal: DeviceHAL) {
private val TAG = "KaleidoscopeEngine"
private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
private var running = false
private val mapper = NeuralHardwareMapper(hal.ctx, 2000f)
