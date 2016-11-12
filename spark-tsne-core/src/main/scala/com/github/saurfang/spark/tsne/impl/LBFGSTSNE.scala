package com.github.saurfang.spark.tsne.impl

import breeze.linalg._
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS}
import breeze.stats.distributions.Rand
import com.github.saurfang.spark.tsne.{TSNEGradient, X2P}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.slf4j.LoggerFactory

import scala.util.Random

/**
 * TODO: This doesn't work at all (yet or ever).
 */
object LBFGSTSNE {
  private def logger = LoggerFactory.getLogger(LBFGSTSNE.getClass)

  def tsne(
            input: RowMatrix,
            noDims: Int = 2,
            maxNumIterations: Int = 1000,
            numCorrections: Int = 10,
            convergenceTol: Double = 1e-4,
            perplexity: Double = 30,
            seed: Long = Random.nextLong()): DenseMatrix[Double] = {
    if(input.rows.getStorageLevel == StorageLevel.NONE) {
      logger.warn("Input is not persisted and performance could be bad")
    }

    Rand.generator.setSeed(seed)

    val n = input.numRows().toInt
    val early_exaggeration = 100
    val t_momentum = 250
    val initial_momentum = 0.5
    val final_momentum = 0.8
    val eta = 500.0
    val min_gain = 0.01

    val Y: DenseMatrix[Double] = DenseMatrix.rand(n, noDims, Rand.gaussian) //:* .0001
    val iY = DenseMatrix.zeros[Double](n, noDims)
    val gains = DenseMatrix.ones[Double](n, noDims)

    // approximate p_{j|i}
    val p_ji = X2P(input, 1e-5, perplexity)
    //logInfo(p_ji.toRowMatrix().rows.collect().toList.toString)
    // p_ij = (p_{i|j} + p_{j|i}) / 2n
    val P = p_ji.transpose().entries.union(p_ji.entries)
      .map(e => ((e.i.toInt, e.j.toInt), e.value))
      .reduceByKey(_ + _)
      .map{case ((i, j), v) => (i, (j, v / 2 / n)) }
      .groupByKey()
      .glom()
      .cache()

      var iteration = 1

      {
        val costFun = new CostFun(P, n, noDims, true)
        val lbfgs = new LBFGS[DenseVector[Double]](maxNumIterations, numCorrections, convergenceTol)
        val states = lbfgs.iterations(new CachedDiffFunction(costFun), new DenseVector(Y.data))

        while (states.hasNext) {
          val state = states.next()
          val loss = state.value
          //logInfo(state.convergedReason.get.toString)
          logger.debug(s"Iteration $iteration finished with $loss")

          Y := asDenseMatrix(state.x, n, noDims)
          //subscriber.onNext((iteration, Y.copy, Some(loss)))
          iteration += 1
        }
      }

      {
        val costFun = new CostFun(P, n, noDims, false)
        val lbfgs = new LBFGS[DenseVector[Double]](maxNumIterations, numCorrections, convergenceTol)
        val states = lbfgs.iterations(new CachedDiffFunction(costFun), new DenseVector(Y.data))

        while (states.hasNext) {
          val state = states.next()
          val loss = state.value
          //logInfo(state.convergedReason.get.toString)
          logger.debug(s"Iteration $iteration finished with $loss")

          Y := asDenseMatrix(state.x, n, noDims)
          //subscriber.onNext((iteration, Y.copy, Some(loss)))
          iteration += 1
        }
      }

      Y
  }

  private[this] def asDenseMatrix(v: DenseVector[Double], n: Int, noDims: Int) = {
    v.asDenseMatrix.reshape(n, noDims)
  }

  private class CostFun(
                         P: RDD[Array[(Int, Iterable[(Int, Double)])]],
                         n: Int,
                         noDims: Int,
                         exaggeration: Boolean) extends DiffFunction[DenseVector[Double]] {

    override def calculate(weights: DenseVector[Double]): (Double, DenseVector[Double]) = {
      val bcY = P.context.broadcast(asDenseMatrix(weights, n, noDims))
      val bcExaggeration = P.context.broadcast(exaggeration)

      val numerator = P.map{ arr => TSNEGradient.computeNumerator(bcY.value, arr.map(_._1): _*) }.cache()
      val bcNumerator = P.context.broadcast({
        numerator.treeAggregate(0.0)(seqOp = (x, v) => x + sum(v), combOp = _ + _)
      })

      val (dY, loss) = P.zip(numerator).treeAggregate((DenseMatrix.zeros[Double](n, noDims), 0.0))(
        seqOp = (c, v) => {
          // c: (grad, loss), v: (Array[(i, Iterable(j, Distance))], numerator)
          // TODO: See if we can include early_exaggeration
          val l = TSNEGradient.compute(v._1, bcY.value, v._2, bcNumerator.value, c._1, bcExaggeration.value)
          (c._1, c._2 + l)
        },
        combOp = (c1, c2) => {
          // c: (grad, loss)
          (c1._1 += c2._1, c1._2 + c2._2)
        })

      numerator.unpersist()

      (loss, new DenseVector(dY.data))
    }
  }
}
