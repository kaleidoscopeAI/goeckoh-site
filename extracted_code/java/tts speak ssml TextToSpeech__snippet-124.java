              val fullShort = ShortArray(samples.size)
              for (i in samples.indices) {
                  fullShort[i] = (samples[i] * Short.MAX_VALUE).toShort()
              }
              ByteBuffer.wrap(byteArray).asShortBuffer().put(fullShort)
              audioBuffer.write(byteArray)
          }
      })

      recorder.start()
  }

  private fun logVadMetrics() {
      val avgLatency = if (totalFrames > 0) vadLatencySum / totalFrames
