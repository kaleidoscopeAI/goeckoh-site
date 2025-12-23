private val E: DoubleArray = DoubleArray(nNodes) { Random.nextDouble(-0.5, 0.5) }
private val W: RealMatrix = Array2DRowRealMatrix(nNodes, nNodes).apply {
    for (i in 0 until nNodes) {
        for (j in 0 until nNodes) {
            setEntry(i, j, if (Random.nextDouble() < 0.1) Random.nextDouble(-0.2, 0.2) else 0.0)
        }
    }
}
private val decayRate = 0.05
private val diffusionRate = 0.1
private val noiseLevel = 0.01
var gcl: Double = 0.5 // Public for UI

private inner class HeartODE(private val externalInput: DoubleArray) : FirstOrderDifferentialEquations {
    override fun getDimension(): Int = nNodes
    override fun computeDerivatives(t: Double, y: DoubleArray, yDot: DoubleArray) {
        val diffusion = W.multiply(Array2DRowRealMatrix(y)).scalarMultiply(diffusionRate)
        for (i in 0 until nNodes) {
            yDot[i] = -decayRate * y[i] + diffusion.getEntry(i, 0) + externalInput[i] + Random.nextDouble(-noiseLevel, noiseLevel)
        }
    }
}

private val integrator = DormandPrince853Integrator(1e-8, 100.0, 1e-10, 1e-10)
private val externalInput = DoubleArray(nNodes) { 0.0 }

fun updateAndGetGCL(stimulus: Double, volume: Double): Double {
    // Apply stimulus modulated by volume
    val effectiveStim = stimulus * (1 + volume)
    for (i in 0 until nNodes) {
        externalInput[i] = effectiveStim * exp(-i.toDouble() / (nNodes / 10.0)) * sin(i.toDouble() / 10.0)
    }

    val yOut = DoubleArray(nNodes)
    integrator.integrate(HeartODE(externalInput), 0.0, E, 1.0, yOut)
    yOut.copyInto(E)

    // GCL: Coherence as correlation + low variance
    val mean = E.average()
    val variance = E.map { (it - mean).pow(2) }.average()
    val coherence = calculateGlobalCoherence(E)
    gcl = (1.0 / (1.0 + variance) + coherence) / 2.0
    return gcl
}

private fun calculateGlobalCoherence(states: DoubleArray): Double {
    // Simple average pairwise correlation approximation
    var sumCorr = 0.0
    val count = 100 // Sample pairs to avoid O(n^2)
    for (k in 0 until count) {
        val i = Random.nextInt(nNodes)
        val j = Random.nextInt(nNodes)
        if (i != j) sumCorr += (states[i] * states[j]).coerceIn(-1.0, 1.0)
    }
    return (sumCorr / count + 1.0) / 2.0 // Normalized 0-1
}
