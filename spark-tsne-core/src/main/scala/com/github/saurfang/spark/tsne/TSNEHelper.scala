package com.github.saurfang.spark.tsne

import breeze.linalg._
import breeze.stats._
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import org.apache.spark.rdd.RDD

object TSNEHelper {
  // p_ij = (p_{i|j} + p_{j|i}) / 2n
  def computeP(p_ji: CoordinateMatrix, n: Int): RDD[(Int, Iterable[(Int, Double)])] = {
    p_ji.entries
      .flatMap(e => Seq(
      ((e.i.toInt, e.j.toInt), e.value),
      ((e.j.toInt, e.i.toInt), e.value)
    ))
      .reduceByKey(_ + _) // p + p'
      .map{case ((i, j), v) => (i, (j, math.max(v / 2 / n, 1e-12))) } // p / 2n
      .groupByKey()
  }

  /**
   * Update Y via gradient dY
   * @param Y current Y
   * @param dY gradient dY
   * @param iY stored y_i - y_{i-1}
   * @param gains adaptive learning rates
   * @param iteration n
   * @param param [[TSNEParam]]
   * @return
   */
  def update(Y: DenseMatrix[Double],
             dY: DenseMatrix[Double],
             iY: DenseMatrix[Double],
             gains: DenseMatrix[Double],
             iteration: Int,
             param: TSNEParam): DenseMatrix[Double] = {
    import param._
    val momentum = if (iteration <= t_momentum) initial_momentum else final_momentum
    gains.foreachPair {
      case ((i, j), old_gain) =>
        val new_gain = math.max(min_gain,
          if ((dY(i, j) > 0.0) != (iY(i, j) > 0.0))
            old_gain + 0.2
          else
            old_gain * 0.8
        )
        gains.update(i, j, new_gain)

        val new_iY = momentum * iY(i, j) - eta * new_gain * dY(i, j)
        iY.update(i, j, new_iY)

        Y.update(i, j, Y(i, j) + new_iY) // Y += iY
    }
    val t_Y: DenseVector[Double] = mean(Y(::, *)).t
    val y_sub = Y(*, ::)
    Y := y_sub - t_Y
  }
}
