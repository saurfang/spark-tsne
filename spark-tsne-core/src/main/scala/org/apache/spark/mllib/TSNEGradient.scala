package org.apache.spark.mllib

import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors}
import org.apache.spark.mllib.optimization.Gradient

class TSNEGradient extends Gradient {
  //override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = ???

  override def compute(
                        data: linalg.Vector,
                        label: Double,
                        weights: linalg.Vector,
                        cumGradient: linalg.Vector): Double = {

  }
}
