package com.github.saurfang.spark.tsne.impl

import breeze.linalg._
import breeze.stats.distributions.Rand
import com.github.saurfang.spark.tsne.tree.SPTree
import com.github.saurfang.spark.tsne.{TSNEGradient, TSNEHelper, TSNEParam, X2P}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.storage.StorageLevel
import org.slf4j.LoggerFactory

import scala.util.Random

object BHTSNE {
  private def logger = LoggerFactory.getLogger(BHTSNE.getClass)

  def tsne(
            input: RowMatrix,
            noDims: Int = 2,
            maxIterations: Int = 1000,
            perplexity: Double = 30,
            theta: Double = 0.5,
            reportLoss: Int => Boolean = {i => i % 10 == 0},
            callback: (Int, DenseMatrix[Double], Option[Double]) => Unit = {case _ => },
            seed: Long = Random.nextLong()
            ): DenseMatrix[Double] = {
    if(input.rows.getStorageLevel == StorageLevel.NONE) {
      logger.warn("Input is not persisted and performance could be bad")
    }

    Rand.generator.setSeed(seed)

    val tsneParam = TSNEParam()
    import tsneParam._

    val n = input.numRows().toInt
    val Y: DenseMatrix[Double] = DenseMatrix.rand(n, noDims, Rand.gaussian(0, 1)) :/ 1e4
    val iY = DenseMatrix.zeros[Double](n, noDims)
    val gains = DenseMatrix.ones[Double](n, noDims)

    // approximate p_{j|i}
    val p_ji = X2P(input, 1e-5, perplexity)
    val P = TSNEHelper.computeP(p_ji, n).glom()
      .map(rows => rows.map {
      case (i, data) =>
        (i, data.map(_._1).toSeq, DenseVector(data.map(_._2 * exaggeration_factor).toArray))
    })
      .cache()

      var iteration = 1
      while(iteration <= maxIterations) {
        val bcY = P.context.broadcast(Y)
        val bcTree = P.context.broadcast(SPTree(Y))

        val initialValue = (DenseMatrix.zeros[Double](n, noDims), DenseMatrix.zeros[Double](n, noDims), 0.0)
        val (posF, negF, sumQ) = P.treeAggregate(initialValue)(
          seqOp = (c, v) => {
            // c: (pos, neg, sumQ), v: Array[(i, Seq(j), vec(Distance))]
            TSNEGradient.computeEdgeForces(v, bcY.value, c._1)
            val q = TSNEGradient.computeNonEdgeForces(bcTree.value, bcY.value, theta, c._2, v.map(_._1): _*)
            (c._1, c._2, c._3 + q)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss)
            (c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })
        val dY: DenseMatrix[Double] = posF :- (negF :/ sumQ)

        TSNEHelper.update(Y, dY, iY, gains, iteration, tsneParam)

        if(reportLoss(iteration)) {
          val loss = P.treeAggregate(0.0)(
            seqOp = (c, v) => {
              TSNEGradient.computeLoss(v, bcY.value, sumQ)
            },
            combOp = _ + _
          )
          logger.debug(s"Iteration $iteration finished with $loss")
          callback(iteration, Y.copy, Some(loss))
        } else {
          logger.debug(s"Iteration $iteration finished")
          callback(iteration, Y.copy, None)
        }

        bcY.destroy()
        bcTree.destroy()

        //undo early exaggeration
        if(iteration == early_exaggeration) {
          P.foreach {
            rows => rows.foreach {
              case (_, _, vec) => vec.foreachPair { case (i, v) => vec.update(i, v / exaggeration_factor) }
            }
          }
        }

        iteration += 1
      }

    Y
  }
}
