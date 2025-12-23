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
