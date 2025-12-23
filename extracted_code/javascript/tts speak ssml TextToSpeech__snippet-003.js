      recognizer.setRecognitionListener(object : RecognitionListener {
          override fun onPartialResult(hypothesis: String?) {}
          override fun onResult(hypothesis: String?) {
              val rawText = hypothesis?.let { parseJsonResult(it) } ?:
