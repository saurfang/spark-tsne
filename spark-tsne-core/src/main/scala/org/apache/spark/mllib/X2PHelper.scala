package org.apache.spark.mllib

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
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

}
