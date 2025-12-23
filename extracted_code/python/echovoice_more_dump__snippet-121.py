package com.kaleidoscope.body
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
data class ControlVector(val cpuFreq: Float, val displayGamma: Float, val networkQos: FloatArray)
class NeuralHardwareMapper(private val maxFreq: Float = 2000f) {
