      var index = 0
      dispatcher.addAudioProcessor(object : AudioProcessor {
          override fun process(audioEvent: AudioEvent): Boolean {
              val buffer = audioEvent.floatBuffer
              buffer.copyInto(processedBuffer, index)
              index += buffer.size
              return true
          }
          override fun processingFinished() {}
      })
      dispatcher.run()

      val processedSamples = processedBuffer.copyOf(index)

      // Play the processed audio
      playAudio(processedSamples, ttsSampleRate)
  }

  private fun playAudio(samples: FloatArray, sampleRate: Int) {
      val bufferSize = AudioTrack.getMinBufferSize(sampleRate,
