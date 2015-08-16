package com.github.saurfang.spark.tsne

import breeze.linalg._
import breeze.stats._
import breeze.stats.distributions.Rand
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.storage.StorageLevel
import rx.lang.scala.Observable

import scala.util.Random

object SimpleTSNE extends Logging {
  def tsne(
            input: RowMatrix,
            noDims: Int = 2,
            maxIterations: Int = 1000,
            perplexity: Double = 30,
            seed: Long = Random.nextLong()): Observable[(Int, DenseMatrix[Double], Double)] = {
    if(input.rows.getStorageLevel == StorageLevel.NONE) {
      logWarning("Input is not persisted and performance could be bad")
    }

    val n = input.numRows().toInt
    val early_exaggeration = 100
    val t_momentum = 20
    val initial_momentum = 0.5
    val final_momentum = 0.8
    val eta = 500.0
    val min_gain = 0.01

    val Y: DenseMatrix[Double] = DenseMatrix.rand(n, noDims, Rand.gaussian) :* .0001
    val iY = DenseMatrix.zeros[Double](n, noDims)
    val gains = DenseMatrix.ones[Double](n, noDims)

    // approximate p_{j|i}
    val p_ji = X2P(input, 1e-5, perplexity)
    //logInfo(p_ji.toRowMatrix().rows.collect().toList.toString)
    // p_ij = (p_{i|j} + p_{j|i}) / 2n
    val p_sum = p_ji.entries
      .flatMap(e => Seq(
        ((e.i.toInt, e.j.toInt), e.value),
        ((e.j.toInt, e.i.toInt), e.value)
      ))
      .reduceByKey(_ + _)
      .cache()
    val p_total = p_sum.map(_._2).sum()
    val P = p_sum
      .map{case ((i, j), v) => (i, (j, max(v / p_total, 1e-12))) }
      .groupByKey()
      .glom()
      .cache()
    p_sum.unpersist()

    Observable(subscriber => {
      var iteration = 1
      while(iteration <= maxIterations && !subscriber.isUnsubscribed) {
        val bcY = P.context.broadcast(Y)

        val numerator = P.map{ arr => TSNEGradient.computeNumerator(bcY.value, arr.map(_._1): _*) }.cache()
        val bcNumerator = P.context.broadcast({
          numerator.treeAggregate(0.0)(seqOp = (x, v) => x + sum(v), combOp = _ + _)
        })

        val (dY, loss) = P.zip(numerator).treeAggregate((DenseMatrix.zeros[Double](n, noDims), 0.0))(
          seqOp = (c, v) => {
            // c: (grad, loss), v: (Array[(i, Iterable(j, Distance))], numerator)
            val l = TSNEGradient.compute(v._1, bcY.value, v._2, bcNumerator.value, c._1, iteration < early_exaggeration)
            (c._1, c._2 + l)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss)
            (c1._1 + c2._1, c1._2 + c2._2)
          })

        bcY.destroy()
        numerator.unpersist()

        val momentum = if (iteration <= t_momentum) initial_momentum else final_momentum
        val dYiY = (dY :> 0.0) :!= (iY :> 0.0)
        gains.foreachPair{
          case ((i, j), gain) =>
            gains.unsafeUpdate(i, j, math.max(min_gain, if(dYiY(i, j)) gain + 0.2 else gain * 0.8))
        }
        iY := (momentum :* iY) - (eta :* gains :* dY)
        Y :+= iY
        Y := Y(*, ::) - (mean(Y(::, *)): DenseMatrix[Double]).toDenseVector

        logDebug(s"Iteration $iteration finished with $loss")
        subscriber.onNext((iteration, Y.copy, loss))
        iteration += 1
      }
      if(!subscriber.isUnsubscribed) subscriber.onCompleted()
    })
  }
}
