  }
  private val decayRate = 0.1
  private val diffusionRate = 0.05

  // ODE definition
  private inner class HeartODE : FirstOrderDifferentialEquations {
      override fun getDimension(): Int = nNodes
      override fun computeDerivatives(t: Double, y: DoubleArray, yDot:
