package org.apache.spark.mllib

import org.apache.spark.mllib.linalg.BLAS.dot
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils


object X2PHelper {

  def fastSquaredDistance(v1: VectorWithNorm, v2: VectorWithNorm): Double = {
    MLUtils.fastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm)
  }

  class VectorWithNorm(val vector: Vector, val norm: Double) extends Serializable {
    def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))

    def this(array: Array[Double]) = this(Vectors.dense(array))

    /** Converts the vector to a dense vector. */
    def toDense: VectorWithNorm = new VectorWithNorm(Vectors.dense(vector.toArray), norm)
  }

  def Hbeta(D: Vector, beta: Double = 1.0) : (Double, Array[Double]) = {
    val P = Vectors.dense(D.toArray.map(d => math.exp(-d * beta)))
    val sumP = P.toArray.sum
    if(sumP == 0) {
      (0.0, Vectors.zeros(D.size).toArray)
    }else {
      val H = Math.log(sumP) + beta * dot(D, P) / sumP
      (H, P.toArray.map(_ / sumP))
    }
  }
}
