// - For full functionality, download Vosk models from https://alphacephei.com/vosk/models and place in assets.

package com.exocortex.neuroacoustic

import android.app.Application
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import be.tarsos.dsp.AudioDispatcher
import be.tarsos.dsp.AudioEvent
import be.tarsos.dsp.pitch.PitchDetectionHandler
import be.tarsos.dsp.pitch.PitchDetectionResult
import be.tarsos.dsp.pitch.PitchProcessor
import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations
import org.apache.commons.math3.ode.nonstiff.DormandPrince853Integrator
import org.kaldi.Model
import org.kaldi.RecognitionListener
import org.kaldi.Vosk
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.sin
import kotlin.random.Random

// Global Application class for initialization
class ExocortexApplication : Application() {
    companion object {
        lateinit var instance: ExocortexApplication
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        Vosk.init(this) // Initialize Vosk
    }
