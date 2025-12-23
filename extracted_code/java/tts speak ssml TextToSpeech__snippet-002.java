class CrystallineHeart(private val nNodes: Int) {
    private val E: DoubleArray = DoubleArray(nNodes) { Random.nextDouble(-1.0, 1.0) } // Node states
    private val W: RealMatrix = Array2DRowRealMatrix(nNodes, nNodes).apply {
        // Random connectivity graph
        for (i in 0 until nNodes) for (j in 0 until nNodes) setEntry(i, j, if (Random.nextBoolean()) 0.1 else 0.0)
    }
    private val decayRate = 0.1
    private val diffusionRate = 0.05

    // ODE definition
    private inner class HeartODE : FirstOrderDifferentialEquations {
        override fun getDimension(): Int = nNodes
        override fun computeDerivatives(t: Double, y: DoubleArray, yDot: DoubleArray) {
            val diffusion = W.multiply(Array2DRowRealMatrix(y)).scalarMultiply(diffusionRate)
            for (i in 0 until nNodes) {
                yDot[i] = -decayRate * y[i] + diffusion.getEntry(i, 0) + externalInput[i]
            }
        }
    }

    private val integrator = DormandPrince853Integrator(1e-8, 100.0, 1e-10, 1e-10)
    private val externalInput = DoubleArray(nNodes) { 0.0 }

    // Update with external stimulus (e.g., vocal arousal)
    fun updateAndGetGCL(stimulus: Double): Double {
        // Apply stimulus to nodes
        for (i in 0 until nNodes) externalInput[i] = stimulus * exp(-i.toDouble() / nNodes)

        // Integrate ODE over time step (e.g., dt=1.0)
        val yOut = DoubleArray(nNodes)
        integrator.integrate(HeartODE(), 0.0, E, 1.0, yOut)
        yOut.copyInto(E)

        // Compute GCL: Average coherence (e.g., inverse variance + harmony)
        val mean = E.average()
        val variance = E.map { (it - mean) * (it - mean) }.average()
        return 1.0 / (1.0 + variance) // Normalized 0-1, high coherence = high GCL
    }
