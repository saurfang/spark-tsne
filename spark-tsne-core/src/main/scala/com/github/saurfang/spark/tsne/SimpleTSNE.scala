package com.github.saurfang.spark.tsne

import org.apache.spark.Logging
import org.apache.spark.mllib.{TSNEGradient, linalg}
import org.apache.spark.mllib.linalg.{Vectors, DenseMatrix}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.mllib.optimization.{Gradient, SimpleUpdater, SquaredL2Updater, LBFGS}
import org.apache.spark.storage.StorageLevel

object SimpleTSNE extends Logging {
  def tsne(x: RowMatrix, noDims: Int = 2, pcaDims: Int = -1, perplexity: Double = 30) = {
    if(x.rows.getStorageLevel != StorageLevel.NONE) {
      logWarning("Input is not persisted and performance could be bad")
    }

    val input = if(pcaDims > 0) x.multiply(x.computePrincipalComponents(pcaDims)) else x
    val n = input.numRows()

    // approximate p_{j|i}
    val p_ji = X2P(input, 1e-5, perplexity)
    // p_ij = (p_{i|j} + p_{j|i}) / 2n
    val p_ij = p_ji.transpose().entries.union(p_ji.entries)
      .map(e => ((e.i, e.j), e.value))
      .reduceByKey(_ + _)
      // early exaggeration
      .map{case ((i, j), v) => MatrixEntry(i, j, v / 2 / n * 4)}

    val p = new CoordinateMatrix(
      X2P(input, 1e-5, perplexity).entries.map(e => e.copy(value = e.value * 4))
    )

    //
  }
}



class TSNEWithLBFGS extends Serializable {
  val optimizer = new LBFGS(new TSNEGradient, new SimpleUpdater)
}