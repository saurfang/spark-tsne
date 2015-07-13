package com.github.saurfang.spark.tsne.examples


import java.io.{OutputStreamWriter, BufferedWriter}

import com.github.saurfang.spark.tsne.SimpleTSNE
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.{Logging, SparkConf, SparkContext}

object MNIST extends Logging {
  def main (args: Array[String]) {
    val sc = new SparkContext(new SparkConf)
    val hadoopConf = sc.hadoopConfiguration
    val fs = FileSystem.get(hadoopConf)

    val dataset = sc.textFile("data/MNIST/train.csv", 10)
      .sample(false, 0.1)
      .filter(!_.startsWith("\""))
      .map(x => x.split(","))
      .map(x => (x.head.toInt, x.tail.map(_.toDouble)))
      .cache()

    val labels = dataset.map(_._1).collect()
    val matrix = new RowMatrix(dataset.map(x => Vectors.dense(x._2)))
    val pcaMatrix = matrix.multiply(matrix.computePrincipalComponents(100))
    pcaMatrix.rows.cache()

    val costWriter = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(s".tmp/MNIST/cost.txt"), true)))

    SimpleTSNE.tsne(pcaMatrix, maxIterations = 500).subscribe {
      res =>
        val (i, y, loss) = res
          logInfo(s"$i iteration finished with loss $loss")

        val os = fs.create(new Path(s".tmp/MNIST/result${"%05d".format(i)}.csv"), true)
        val writer = new BufferedWriter(new OutputStreamWriter(os))
        try {
          (0 until y.rows).foreach {
            row =>
              writer.write(labels(row).toString)
              writer.write(y(row, ::).inner.toArray.mkString(",", ",", "\n"))
          }
          costWriter.write(loss + "\n")
        } finally {
          writer.close()
        }
    }
    costWriter.close()

    sc.stop()
  }
}
