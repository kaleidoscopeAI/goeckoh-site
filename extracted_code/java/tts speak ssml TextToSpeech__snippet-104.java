                      val correctedText = correctToFirstPerson(rawText)
                      speakCorrectedText(correctedText, prosody)
                      callback(correctedText, prosody)
                      break
                  }
              } else {
                  silenceStart = 0L
              }
          } else if (read < 0) {
              break
          }
      }

      audioRecord?.stop()
      audioRecord?.release()
  }

  private fun parseResultWithConfidence(json: String): Pair<String, Double>
