      }
  }

  private fun initSystem() {
      // Check for LLM model
      val modelFile = File(filesDir, "llm/model.task")
      if (!modelFile.exists()) {
          statusText.text = "LLM model missing. Download Gemma-3 1B and
