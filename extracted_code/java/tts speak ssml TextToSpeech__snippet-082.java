                      recognizer.result
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

  private fun parseJsonResult(json: String): String {
      val textStart = json.indexOf("\"text\" : \"") + 10
      val textEnd = json.lastIndexOf("\"")
      return if (textStart > 9 && textEnd > textStart)
