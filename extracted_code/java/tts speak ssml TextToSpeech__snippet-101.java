      }
  }

  private fun initSystem() {
      val ttsModels = listOf(
          "vits-piper-en_US-amy-medium.onnx" to "en",
          "vits-piper-es_ES-mls_9972-medium.onnx" to "es",
          "vits-piper-fr_FR-upmc-medium.onnx" to "fr"
      )
      ttsModels.forEach { (modelName, lang) ->
          val ttsModelFile = File(filesDir, modelName)
          if (!ttsModelFile.exists()) {
              try {
                  assets.open(modelName).use { input ->
                      FileOutputStream(ttsModelFile).use { output ->
                          input.copyTo(output)
                      }
                  }
              } catch (e: Exception) {
                  statusText.text = "TTS model copy failed for $lang:
