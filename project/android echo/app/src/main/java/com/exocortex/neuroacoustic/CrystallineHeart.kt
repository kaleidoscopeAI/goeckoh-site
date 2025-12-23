package com.exocortex.neuroacoustic

import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations
import org.apache.commons.math3.ode.nonstiff.DormandPrince853Integrator
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sin
import kotlin.random.Random

class CrystallineHeart(private val nNodes: Int) {
    private val E: DoubleArray = DoubleArray(nNodes) { Random.nextDouble(-0.5, 0.5) }
    private val W: RealMatrix = Array2DRowRealMatrix(nNodes, nNodes).apply {
        for (i in 0 until nNodes) {
            for (j in 0 until nNodes) {
                setEntry(i, j, if (Random.nextDouble() < 0.05) Random.nextDouble(-0.3, 0.3) else 0.0)
            }
        }
    }
    private val decayRate = 0.05
    private val diffusionRate = 0.1
    private val noiseLevel = 0.01
    var gcl: Double = 0.5
        private set

    private inner class HeartODE(private val externalInput: DoubleArray) : FirstOrderDifferentialEquations {
        override fun getDimension(): Int = nNodes
        override fun computeDerivatives(t: Double, y: DoubleArray, yDot: DoubleArray) {
            val diffusion = W.multiply(Array2DRowRealMatrix(y)).scalarMultiply(diffusionRate)
            for (i in 0 until nNodes) {
                yDot[i] = -decayRate * y[i] + diffusion.getEntry(i, 0) + externalInput[i] + Random.nextGaussian() * noiseLevel
            }
        }
    }

    private val integrator = DormandPrince853Integrator(1e-8, 100.0, 1e-10, 1e-10)
    private val externalInput = DoubleArray(nNodes) { 0.0 }

    fun updateAndGetGCL(stimulus: Double, volume: Double, variance: Double): Double {
        val effectiveStim = stimulus * volume * (1 + variance / 50.0)
        for (i in 0 until nNodes) {
            externalInput[i] = effectiveStim * exp(-i.toDouble() / (nNodes / 5.0)) * sin(2 * Math.PI * i / nNodes)
        }

        val yOut = DoubleArray(nNodes)
        integrator.integrate(HeartODE(externalInput), 0.0, E, 1.0, yOut)
        yOut.copyInto(E)

        val mean = E.average()
        val varStat = E.map { (it - mean).pow(2) }.average()
        val coherence = calculateGlobalCoherence(E)
        gcl = ((1.0 / (1.0 + varStat)) + coherence) / 2.0.coerceIn(0.0, 1.0)
        return gcl
    }

    private fun calculateGlobalCoherence(states: DoubleArray): Double {
        var sumCorr = 0.0
        val sampleSize = minOf(500, nNodes * (nNodes - 1) / 2)
        repeat(sampleSize) {
            val i = Random.nextInt(nNodes)
            var j = Random.nextInt(nNodes)
            while (j == i) j = Random.nextInt(nNodes)
            sumCorr += states[i] * states[j]
        }
        return (sumCorr / sampleSize + 1.0) / 2.0
    }
}
