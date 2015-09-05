package com.github.saurfang.spark.tsne.impl

import breeze.linalg._
import breeze.stats._
import breeze.stats.distributions.Rand
import com.github.saurfang.spark.tsne.tree.SPTree
import com.github.saurfang.spark.tsne.{TSNEGradient, TSNEHelper, X2P, TSNEParam}
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.storage.StorageLevel
import rx.lang.scala.Observable

import scala.util.Random

object BHTSNE extends Logging {
  def tsne(
            input: RowMatrix,
            noDims: Int = 2,
            maxIterations: Int = 1000,
            perplexity: Double = 30,
            theta: Double = 0.5,
            reportLoss: Int => Boolean = {i => i % 10 == 0},
            seed: Long = Random.nextLong()
            ): Observable[(Int, DenseMatrix[Double], Option[Double])] = {
    if(input.rows.getStorageLevel == StorageLevel.NONE) {
      logWarning("Input is not persisted and performance could be bad")
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

    Observable(subscriber => {
      var iteration = 1
      while(iteration <= maxIterations && !subscriber.isUnsubscribed) {
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
          logDebug(s"Iteration $iteration finished with $loss")
          subscriber.onNext((iteration, Y.copy, Some(loss)))
        } else {
          logDebug(s"Iteration $iteration finished")
          subscriber.onNext((iteration, Y.copy, None))
        }

        bcY.destroy()
        bcTree.destroy()

        //undo early exaggeration
        if(iteration == early_exaggeration) {
          P.foreach {
            rows => rows.foreach {
              case (_, _, vec) => vec.foreachPair { case (i, v) => vec.unsafeUpdate(i, v / exaggeration_factor) }
            }
          }
        }

        iteration += 1
      }
      if(!subscriber.isUnsubscribed) subscriber.onCompleted()
    })
  }
}
