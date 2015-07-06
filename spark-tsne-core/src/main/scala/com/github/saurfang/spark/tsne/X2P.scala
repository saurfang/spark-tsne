package com.github.saurfang.spark.tsne

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{Vectors, Matrices}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, RowMatrix, DistributedMatrix}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.X2PHelper._
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._

/**
 * X2P Identifies appropriate sigma's to get kk NNs up to some tolerance
 *
 *    [P, beta] = x2p(xx, kk, tol)
 *
 *  Identifies the required precision (= 1 / variance^2) to obtain a Gaussian
 *  kernel with a certain uncertainty for every datapoint. The desired
 *  uncertainty can be specified through the perplexity u (default = 15). The
 *  desired perplexity is obtained up to some tolerance that can be specified
 *  by tol (default = 1e-4).
 *  The function returns the final Gaussian kernel in P, as well as the
 *  employed precisions per instance in beta.
 *
 *  https://github.com/lvdmaaten/lvdmaaten.github.io/tree/master/tsne/code
 */
object X2P extends Logging {
  def apply(x: RowMatrix, tol: Double = 1e-4, perplexity: Int = 15) = {
    val n = x.numRows()
    val mu = 3 * perplexity
    val norms = x.rows.map(Vectors.norm(_, 2.0))
    val rowsWithNorm = x.rows.zip(norms).map{ case (v, norm) => new VectorWithNorm(v, norm) }
    val neighbors = rowsWithNorm
      .zipWithIndex()
      .flatMap{ case (v, i) => (1L to n).filter(_ != i).map(j => (j, v))}
      .join(rowsWithNorm.zipWithIndex().map{case (v, i) => (i, v)})
      .map{ case (i, (u, v)) => (i, fastSquaredDistance(u, v)) }
      .topByKey(mu)
  }
}
