if "class NeuralHardwareMapper(private val ctx: Context" not in nhm_text:
nhm_text = nhm_text.replace("class NeuralHardwareMapper(private val maxFreq: Float = 2000f) {",
"import android.content.Context\n\nclass NeuralHardwareMapper(private val ctx: Context, private val maxFreq:
