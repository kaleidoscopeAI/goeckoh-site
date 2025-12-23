 1 +package com.exocortex.neuroacoustic
 2 +
 3 +import android.content.Context
 4 +import android.widget.TextView
 5 +import java.io.File
 6 +import java.io.FileOutputStream
 7 +import java.util.zip.ZipInputStream
 8 +
 9 +object AssetInstaller {
10 +    private val ttsModels = listOf(
11 +        "vits-piper-en_US-amy-medium.onnx",
12 +        "vits-piper-es_ES-mls_9972-medium.onnx",
13 +        "vits-piper-fr_FR-upmc-medium.onnx"
14 +    )
15 +
16 +    private val voskZips = listOf(
17 +        "vosk-model-small-en-us-0.15.zip" to "vosk-model-small-en-us",
18 +        "vosk-model-small-es-0.42.zip" to "vosk-model-small-es",
19 +        "vosk-model-small-fr-0.22.zip" to "vosk-model-small-fr"
20 +    )
21 +
22 +    fun installAll(context: Context, statusView: TextView? = null) {
23 +        copyTts(context, statusView)
24 +        unzipVosk(context, statusView)
25 +        ensureLlmmodel(context, statusView)
26 +    }
27 +
28 +    private fun copyTts(context: Context, statusView: TextView?) {
29 +        ttsModels.forEach { model ->
30 +            val outFile = File(context.filesDir, model)
31 +            if (outFile.exists()) return@forEach
32 +            statusView?.text = "Copying $model..."
33 +            context.assets.open(model).use { input ->
34 +                FileOutputStream(outFile).use { output -> input.copyTo(o
    utput) }
35 +            }
36 +        }
37 +    }
38 +
39 +    private fun unzipVosk(context: Context, statusView: TextView?) {
40 +        voskZips.forEach { (zipName, dirName) ->
41 +            val outDir = File(context.filesDir, dirName)
42 +            if (outDir.exists()) return@forEach
43 +            statusView?.text = "Unpacking $zipName..."
44 +            outDir.mkdirs()
45 +            context.assets.open(zipName).use { input ->
46 +                ZipInputStream(input).use { zip ->
47 +                    var entry = zip.nextEntry
48 +                    while (entry != null) {
49 +                        val outFile = File(outDir, entry.name)
50 +                        if (entry.isDirectory) {
51 +                            outFile.mkdirs()
52 +                        } else {
53 +                            outFile.parentFile?.mkdirs()
54 +                            FileOutputStream(outFile).use { output -> zi
    p.copyTo(output) }
55 +                        }
56 +                        entry = zip.nextEntry
57 +                    }
58 +                }
59 +            }
60 +        }
61 +    }
62 +
63 +    private fun ensureLlmmodel(context: Context, statusView: TextView?)
    {
64 +        val llm = File(context.filesDir, "gemma-1.1-2b-it-q4f16.task")
65 +        if (!llm.exists()) {
66 +            statusView?.text = "LLM missing: place gemma-1.1-2b-it-q4f16
    .task in filesDir"
67 +        }
68 +    }
69 +}

