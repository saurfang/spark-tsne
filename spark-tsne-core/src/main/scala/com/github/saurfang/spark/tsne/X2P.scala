package com.github.saurfang.spark.tsne

import breeze.linalg.sum
import breeze.numerics.exp
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, CoordinateMatrix, RowMatrix, DistributedMatrix}
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
  def apply(x: RowMatrix, tol: Double = 1e-5, perplexity: Double = 30.0) = {
    require(tol >= 0, "Tolerance must be non-negative")
    require(perplexity > 0, "Perplexity must be positive")

    val n = x.numRows()
    val mu = (10 * perplexity).toInt //TODO: Expose this as parameter
    val logU = Math.log(perplexity)
    val norms = x.rows.map(Vectors.norm(_, 2.0))
    norms.persist()
    val rowsWithNorm = x.rows.zip(norms).map{ case (v, norm) => new VectorWithNorm(v, norm) }
    val neighbors = rowsWithNorm
      .zipWithIndex()
      .flatMap{ case (v, i) => (1L to n).filter(_ != i).map(j => (j, (i, v)))}
      .join(rowsWithNorm.zipWithIndex().map{case (v, i) => (i, v)})
      .map{ case (i, ((j, u), v)) => (i, (j, fastSquaredDistance(u, v))) }
      .filter(_._2._2 > 0)
      .topByKey(mu)(Ordering.by(e => -e._2))
    norms.unpersist()

    new CoordinateMatrix(
      neighbors.flatMap {
        case (i, arr) =>
          var betamin = Double.NegativeInfinity
          var betamax =  Double.PositiveInfinity
          var beta = 1.0

          val d = Vectors.dense(arr.map(_._2))
          var (h, p) = Hbeta(d, beta)

          //logInfo("data was " + d.toArray.toList)
          //logInfo("array P was " + p.toList)

          // Evaluate whether the perplexity is within tolerance
          def Hdiff = h - logU
          var tries = 0
          while (Math.abs(Hdiff) > tol && tries < 50) {
            //If not, increase or decrease precision
            if (Hdiff > 0) {
              betamin = beta
              beta = if (betamax.isInfinite) beta * 2 else (beta + betamax) / 2
            } else {
              betamax = beta
              beta = if (betamin.isInfinite) beta / 2 else (beta + betamin) / 2
            }

            // Recompute the values
            val HP = Hbeta(d, beta)
            h = HP._1
            p = HP._2
            tries = tries + 1
          }

          //logInfo("array P is " + p.toList)

          arr.map(_._1).zip(p).map{ case (j, v) => MatrixEntry(i, j, v) }
      }
    )
  }
}
