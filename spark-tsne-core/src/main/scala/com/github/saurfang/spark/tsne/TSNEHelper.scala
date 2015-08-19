package com.github.saurfang.spark.tsne

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
}
