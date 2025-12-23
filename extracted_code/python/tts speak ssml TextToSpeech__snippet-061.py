          // Here, pseudo-code: check if silence > threshold
          Thread.sleep(100)
          val energy = 0.0 // Compute from recent buffer
          if (energy < 0.01) {
              if (silenceStart == 0L) silenceStart =
