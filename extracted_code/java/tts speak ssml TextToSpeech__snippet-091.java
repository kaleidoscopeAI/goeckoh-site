      }
  }

  private fun initSystem() {
      // Check models
      val ttsModelFile = File(filesDir, "vits-piper-en_US-amy-medium.onnx")
      if (!ttsModelFile.exists()) {
          try {
              assets.open("vits-piper-en_US-amy-medium.onnx").use { input
