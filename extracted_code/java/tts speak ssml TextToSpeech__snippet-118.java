                      recorder.stop()
                      whisper.start() // Finalize
                  }
              }

              // Collect for prosody
              val byteArray = ByteArray(shortArray.size * 2)
              ByteBuffer.wrap(byteArray).asShortBuffer().put(shortArray)
              audioBuffer.write(byteArray)
          }
      })

      recorder.start()
  }

  private fun correctToFirstPerson(text: String): String {
      if (text.isEmpty()) return ""
      var corrected = text.replace(Regex("\\b(you|he|she|they)\\b",
