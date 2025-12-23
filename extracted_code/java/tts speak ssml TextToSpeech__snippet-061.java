  private val externalInput = DoubleArray(nNodes) { 0.0 }

  // Update with external stimulus (e.g., vocal arousal)
  fun updateAndGetGCL(stimulus: Double): Double {
      // Apply stimulus to nodes
      for (i in 0 until nNodes) externalInput[i] = stimulus *
