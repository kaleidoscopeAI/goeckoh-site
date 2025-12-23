                  }
              } else {
                  silenceStart = 0L
              }
          }
      })

      recorder.start()
  }

  private fun correctToFirstPerson(text: String): String {
      if (text.isEmpty()) return ""
      var corrected = text.replace(Regex("\\b(you|he|she|they)\\b",
