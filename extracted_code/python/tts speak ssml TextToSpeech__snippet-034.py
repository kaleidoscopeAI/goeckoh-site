     1 -package com.exocortex.neuroacoustic
     2 -
     3 -import android.content.Context
     4 -import android.media.AudioFormat
     5 -import android.media.AudioRecord
     6 -import android.media.MediaRecorder
     7 -import android.util.Log
     8 -import java.util.concurrent.atomic.AtomicBoolean
     9 -import kotlin.concurrent.thread
    10 -
    11 -interface IRecorderListener {
    12 -    fun onUpdateReceived(message: String)
    13 -    fun onDataReceived(samples: FloatArray)
    14 -}
    15 -
    16 -class Recorder(private val context: Context) {
    17 -    private val sampleRate = 16000
    18 -    private val bufferSize = AudioRecord.getMinBufferSize(
    19 -        sampleRate,
    20 -        AudioFormat.CHANNEL_IN_MONO,
    21 -        AudioFormat.ENCODING_PCM_16BIT
    22 -    )
    23 -    private val isRunning = AtomicBoolean(false)
    24 -    private var thread: Thread? = null
    25 -    private var listener: IRecorderListener? = null
    26 -    private var audioRecord: AudioRecord? = null
    27 -
    28 -    val audioSessionId: Int
    29 -        get() = audioRecord?.audioSessionId ?: 0
    30 -
    31 -    fun setListener(l: IRecorderListener) {
    32 -        listener = l
    33 -    }
    34 -
    35 -    fun start() {
    36 -        if (isRunning.get()) return
    37 -        audioRecord = AudioRecord(
    38 -            MediaRecorder.AudioSource.MIC,
    39 -            sampleRate,
    40 -            AudioFormat.CHANNEL_IN_MONO,
    41 -            AudioFormat.ENCODING_PCM_16BIT,
    42 -            bufferSize
    43 -        )
    44 -        audioRecord?.startRecording()
    45 -        isRunning.set(true)
    46 -        thread = thread(start = true, name = "RecorderThread") {
    47 -            val shortBuffer = ShortArray(bufferSize / 2)
    48 -            while (isRunning.get()) {
    49 -                val read = audioRecord?.read(shortBuffer, 0, shortBuffer
        .size) ?: 0
    50 -                if (read > 0) {
    51 -                    val floatArray = FloatArray(read) { i -> shortBuffer
        [i].toFloat() / Short.MAX_VALUE }
    52 -                    listener?.onDataReceived(floatArray)
    53 -                }
    54 -            }
    55 -        }
    56 -    }
    57 -
    58 -    fun stop() {
    59 -        if (!isRunning.get()) return
    60 -        isRunning.set(false)
    61 -        try {
    62 -            audioRecord?.stop()
    63 -            audioRecord?.release()
    64 -        } catch (_: Exception) {
    65 -        }
    66 -        thread?.join(500)
    67 -        thread = null
    68 -        listener?.onUpdateReceived("Recorder stopped")
    69 -    }
    70 -}

