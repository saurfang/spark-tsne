package org.apache.spark.mllib

import breeze.linalg._
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors}
import org.apache.spark.mllib.optimization.Updater
import breeze.linalg.{norm => brzNorm, axpy => brzAxpy, Vector => BV}
import breeze.numerics._

/**
 * Created by forest on 7/9/15.
 */
class TSNEUpdater extends Updater{
  override def compute(
                        weightsOld: linalg.Vector,
                        gradient: linalg.Vector,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (linalg.Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)

    //first weight is the numerator
    val y = new DenseMatrix(brzWeights.size / 2, 2, brzWeights.toArray.tail).toBreeze
    val sumY = sum(y :* y, Axis._1)
    //num = 1 ./ (1 + ((-2 * (Y * Y')) .+ sum_Y)' .+ sum_Y)
    val numerator = 1 :/ (1 + ((-2 :* y * y.t) + sumY).t + sumY)

    (Vectors.fromBreeze(brzWeights), 0)
  }
}
