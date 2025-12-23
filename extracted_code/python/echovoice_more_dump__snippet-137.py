package com.kaleidoscope.body
import android.content.Context
import android.content.res.AssetFileDescriptor
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
class TFLiteModel(private val ctx: Context) {
private val TAG = "TFLiteModel"
private var interpreter: Interpreter? = null
