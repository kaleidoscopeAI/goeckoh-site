                  vadStartTime = System.currentTimeMillis()
                  val speechProb = sileroVad.process(frame)
                  val latency = System.currentTimeMillis() - vadStartTime
                  vadLatencySum += latency

                  val isSpeech = speechProb > vadThreshold
                  if (isSpeech) {
                      speechFrames++
                      isSpeechDetected = true
                      silenceStart = 0
                  } else if (isSpeechDetected) {
                      if (silenceStart == 0L) silenceStart =
