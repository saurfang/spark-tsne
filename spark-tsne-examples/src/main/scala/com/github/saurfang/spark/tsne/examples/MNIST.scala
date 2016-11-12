package com.github.saurfang.spark.tsne.examples


import java.io.{BufferedWriter, OutputStreamWriter}

import com.github.saurfang.spark.tsne.impl._
import com.github.saurfang.spark.tsne.tree.SPTree
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.LoggerFactory

object MNIST {
  private def logger = LoggerFactory.getLogger(MNIST.getClass)

  def main (args: Array[String]) {
    val conf = new SparkConf()
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .registerKryoClasses(Array(classOf[SPTree]))
    val sc = new SparkContext(conf)
    val hadoopConf = sc.hadoopConfiguration
    val fs = FileSystem.get(hadoopConf)

    val dataset = sc.textFile("data/MNIST/mnist.csv.gz")
      .zipWithIndex()
      .filter(_._2 < 6000)
      .sortBy(_._2, true, 60)
      .map(_._1)
      .map(_.split(","))
      .map(x => (x.head.toInt, x.tail.map(_.toDouble)))
      .cache()
    //logInfo(dataset.collect.map(_._2.toList).toList.toString)

    //val features = dataset.map(x => Vectors.dense(x._2))
    //val scaler = new StandardScaler(true, true).fit(features)
    //val scaledData = scaler.transform(features)
    //  .map(v => Vectors.dense(v.toArray.map(x => if(x.isNaN || x.isInfinite) 0.0 else x)))
    //  .cache()
    val data = dataset.flatMap(_._2)
    val mean = data.mean()
    val std = data.stdev()
    val scaledData = dataset.map(x => Vectors.dense(x._2.map(v => (v - mean) / std))).cache()

    val labels = dataset.map(_._1).collect()
    val matrix = new RowMatrix(scaledData)
    val pcaMatrix = matrix.multiply(matrix.computePrincipalComponents(50))
    pcaMatrix.rows.cache()

    val costWriter = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(s".tmp/MNIST/cost.txt"), true)))

    //SimpleTSNE.tsne(pcaMatrix, perplexity = 20, maxIterations = 200)
    BHTSNE.tsne(pcaMatrix, maxIterations = 500, callback = {
    //LBFGSTSNE.tsne(pcaMatrix, perplexity = 10, maxNumIterations = 500, numCorrections = 10, convergenceTol = 1e-8)
      case (i, y, loss) =>
        if(loss.isDefined) logger.info(s"$i iteration finished with loss $loss")

        val os = fs.create(new Path(s".tmp/MNIST/result${"%05d".format(i)}.csv"), true)
        val writer = new BufferedWriter(new OutputStreamWriter(os))
        try {
          (0 until y.rows).foreach {
            row =>
              writer.write(labels(row).toString)
              writer.write(y(row, ::).inner.toArray.mkString(",", ",", "\n"))
          }
          if(loss.isDefined) costWriter.write(loss.get + "\n")
        } finally {
          writer.close()
        }
    })
    costWriter.close()

    sc.stop()
  }
}
