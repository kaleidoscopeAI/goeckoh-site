      }
  }

  private fun initSystem() {
      neuroAcousticMirror = NeuroAcousticMirror(this, tts)
      crystallineHeart = CrystallineHeart(1024)
      gatedAGI = GatedAGI(crystallineHeart)
      statusText.text = "System Initialized"
  }

  private fun startListening() {
      Executors.newSingleThreadExecutor().execute {
          while (true) {
              neuroAcousticMirror.listenAndProcess { correctedText, prosody
