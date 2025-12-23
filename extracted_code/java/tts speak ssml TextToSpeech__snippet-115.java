                      recorder.stop()
                      whisper.start() // Or call to finalize transcription
                  }
              }

              // Collect for prosody (float to byte)
              audioBuffer.write(byteArray)
          }
      })

      recorder.start()
  }

  private fun correctToFirstPerson(text: String): String {
      if (text.isEmpty()) return ""
      var corrected = text.replace(Regex("\\b(you|he|she|they)\\b",
