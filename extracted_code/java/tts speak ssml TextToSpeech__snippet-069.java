                      recognizer.result
                      break
                  }
              } else {
                  silenceStart = 0L
              }
          }
      }

      audioRecord?.stop()
      audioRecord?.release()
  }

  private fun parseJsonResult(json: String): String {
      return json.substringAfter("\"text\" :
