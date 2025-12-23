  private val W: RealMatrix = Array2DRowRealMatrix(nNodes, nNodes).apply {
      // Random connectivity graph
      for (i in 0 until nNodes) for (j in 0 until nNodes) setEntry(i, j, if
