package com.exocortex.neuroacoustic

import android.content.Context
import android.widget.TextView
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipInputStream

object AssetInstaller {
    private val ttsModels = listOf(
        "vits-piper-en_US-amy-medium.onnx",
        "vits-piper-es_ES-mls_9972-medium.onnx",
        "vits-piper-fr_FR-upmc-medium.onnx"
    )

    private val voskZips = listOf(
        "vosk-model-small-en-us-0.15.zip" to "vosk-model-small-en-us",
        "vosk-model-small-es-0.42.zip" to "vosk-model-small-es",
        "vosk-model-small-fr-0.22.zip" to "vosk-model-small-fr"
    )

    fun installAll(context: Context, statusView: TextView? = null) {
        copyTts(context, statusView)
        unzipVosk(context, statusView)
        ensureLlmmodel(context, statusView)
    }

    private fun copyTts(context: Context, statusView: TextView?) {
        ttsModels.forEach { model ->
            val outFile = File(context.filesDir, model)
            if (outFile.exists()) return@forEach
            statusView?.text = "Copying $model..."
            context.assets.open(model).use { input ->
                FileOutputStream(outFile).use { output -> input.copyTo(output) }
            }
        }
    }

    private fun unzipVosk(context: Context, statusView: TextView?) {
        voskZips.forEach { (zipName, dirName) ->
            val outDir = File(context.filesDir, dirName)
            if (outDir.exists()) return@forEach
            statusView?.text = "Unpacking $zipName..."
            outDir.mkdirs()
            context.assets.open(zipName).use { input ->
                ZipInputStream(input).use { zip ->
                    var entry = zip.nextEntry
                    while (entry != null) {
                        val outFile = File(outDir, entry.name)
                        if (entry.isDirectory) {
                            outFile.mkdirs()
                        } else {
                            outFile.parentFile?.mkdirs()
                            FileOutputStream(outFile).use { output -> zip.copyTo(output) }
                        }
                        entry = zip.nextEntry
                    }
                }
            }
        }
    }

    private fun ensureLlmmodel(context: Context, statusView: TextView?) {
        val llm = File(context.filesDir, "gemma-1.1-2b-it-q4f16.task")
        if (!llm.exists()) {
            statusView?.text = "LLM missing: place gemma-1.1-2b-it-q4f16.task in filesDir"
        }
    }
}
