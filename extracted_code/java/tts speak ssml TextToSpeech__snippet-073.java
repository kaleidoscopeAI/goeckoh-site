  private val W: RealMatrix = Array2DRowRealMatrix(nNodes, nNodes).apply {
      for (i in 0 until nNodes) {
          for (j in 0 until nNodes) {
              setEntry(i, j, if (Random.nextDouble() < 0.1)
