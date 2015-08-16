package com.github.saurfang.spark.tsne

import org.apache.spark.SharedSparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.scalatest.{FunSuite, Matchers}

/**
 * Created by forest on 8/16/15.
 */
class X2PSuite extends FunSuite with SharedSparkContext with Matchers {

  test("Test X2P against tsne.jl implementation") {
    val input = new RowMatrix(
      sc.parallelize(Seq(1 to 3, 4 to 6, 7 to 9, 10 to 12))
        .map(x => Vectors.dense(x.map(_.toDouble).toArray))
    )
    val output = X2P(input, 1e-5, 2).toRowMatrix().rows.collect().map(_.toArray.toList)
    println(output.toList)
    //output shouldBe List(List(0, .5, .5), List(.5, 0, .5), List(.5, .5, .0))
  }
}
