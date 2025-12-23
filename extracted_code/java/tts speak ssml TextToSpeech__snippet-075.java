  private val externalInput = DoubleArray(nNodes) { 0.0 }

  fun updateAndGetGCL(stimulus: Double, volume: Double): Double {
      // Apply stimulus modulated by volume
      val effectiveStim = stimulus * (1 + volume)
      for (i in 0 until nNodes) {
          externalInput[i] = effectiveStim * exp(-i.toDouble() / (nNodes /
