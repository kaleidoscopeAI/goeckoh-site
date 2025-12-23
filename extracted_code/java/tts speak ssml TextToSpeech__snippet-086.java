  }
  private val decayRate = 0.05
  private val diffusionRate = 0.1
  private val noiseLevel = 0.01
  var gcl: Double = 0.5

  private inner class HeartODE(private val externalInput: DoubleArray) :
