package com.github.saurfang.spark.tsne.examples

import com.github.saurfang.spark.tsne.SimpleTSNE
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.{Logging, SparkConf, SparkContext}

/**
 * Created by forest on 7/12/15.
 */
object MNIST extends Logging {
  def main (args: Array[String]) {
    val sc = new SparkContext(new SparkConf)

    val dataset = sc.textFile("data/MNIST/train.csv", 100)
      .filter(!_.startsWith("\""))
      .map(x => x.split(","))
      .map(x => (x.head.toInt, x.tail.map(_.toDouble)))
      .cache()

    val matrix = new RowMatrix(dataset.map(x => Vectors.dense(x._2)))
    SimpleTSNE.tsne(matrix).subscribe {
      res =>
        val (y, loss) = res
        logInfo("pass")
    }
  }
}
