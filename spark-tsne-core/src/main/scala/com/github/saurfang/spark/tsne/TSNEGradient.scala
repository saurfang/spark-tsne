package com.github.saurfang.spark.tsne

import breeze.linalg._
import breeze.numerics._
import org.apache.spark.Logging

object TSNEGradient extends Logging  {
  /**
   *
   * @param i
   * @param Y
   * @return
   */
  def computeNumerator(Y: DenseMatrix[Double], i: Int *): DenseMatrix[Double] = {
    // Y_sum = ||Y_i||^2
    //logInfo(s"i: ${i.toList} \n Y: $Y")
    val sumY = sum((Y :* Y).apply(*, ::))
    val subY = Y(i, ::).toDenseMatrix
    val y1: DenseMatrix[Double] = Y * (-2.0 :* subY.t)
    val num: DenseMatrix[Double] = (y1(::, *) + sumY).t
    num := 1.0 :/ (1.0 :+ (num(::, *) + sumY(i).toDenseVector))

//
//    val y1: DenseMatrix[Double] = -2.0 :* (Y * Y.t)
//    val y2 = (y1(::, *) + sumY).t
//    val num = 1.0 :/ (1.0 :+ (y2(::, *) + sumY))

    // Y_diff = Y_i - Y
//    val yDiff: DenseMatrix[Double] = Y(*, ::) - Y(i, ::).inner
    // Y_sum = ||Y_i - Y||^2
//    val num = sum((yDiff :* yDiff).apply(*, ::))
    //val ySquared = Y :* Y
    //val num = ySquared(i) :- (2 :* Y(i) :* Y) :+ ySquared
    // num = (1 + ||Y_i - Y||^2)^-1
    // num := 1.0 :/ (1.0 :+ num)
    //diag(num) := DenseVector.zeros[Double](i.length)
    i.indices.foreach(idx => num.unsafeUpdate(idx, i(idx), 0.0))
    num := num.mapValues(math.max(_, 1e-12))

    //logInfo(s"num: $num")
    num
  }

  /**
   * Compute the TSNE Gradient at i. Update the gradient through dY then return costs attributed at i.
   *
   * @param data data point for row i by list of pair of (j, p_ij) and 0 <= j < n
   * @param Y current Y [n * 2]
   * @param totalNum the common numerator that captures the t-distribution of Y
   * @param dY gradient of Y
   * @return loss attributed to row i
   */
  def compute(
               data: Array[(Int, Iterable[(Int, Double)])],
               Y: DenseMatrix[Double],
               num: DenseMatrix[Double],
               totalNum: Double,
               dY: DenseMatrix[Double],
               exaggeration: Boolean): Double = {
    //val n = Y.rows
    //val m = data.length
//    val p = {
//      val mat = DenseMatrix.zeros[Double](m, n)
//      data.zipWithIndex.foreach {
//        case ((_, itr), i) =>
//          itr.foreach {
//            case (j, v) =>
//              mat.unsafeUpdate(i, j, math.max(if(exaggeration) v * 4 else v, 1e-12))
//          }
//      }
//      mat
//    }

    // q = (1 + ||Y_i - Y_j||^2)^-1 / sum(1 + ||Y_k - Y_l||^2)^-1
    val q: DenseMatrix[Double] = num :/ totalNum

    val loss = data.zipWithIndex.flatMap{
      case ((_, itr), i) =>
        itr.map{
          case (j, v) =>
            val exaggeratedV = if(exaggeration) v * 4 else v
            val qij = q(i, j)
            val l = exaggeratedV * math.log(exaggeratedV / qij)
            q.unsafeUpdate(i, j,  qij - exaggeratedV)
            if(l.isNaN) 0.0 else l
        }
    }.sum

    // l = [ (p_ij - q_ij) * (1 + ||Y_i - Y_j||^2)^-1 ]
    q := -q :* num
    // l_sum = [0 0 ... sum(l) ... 0]
    //val sumL = DenseMatrix.zeros[Double](m, n)
    sum(q(*, ::)).foreachPair{ case (i, v) => q.unsafeUpdate(i, data(i)._1, q(i, i) - v) }
//    logInfo(s"l : ${l.rows} x ${l.cols}")
//    logInfo(s"sumL : ${sumL.rows} x ${sumL.cols}")
//    logInfo(s"Y : ${Y.rows} x ${Y.cols}")

    // TODO: dY_i = 4 * (l_sum - l) * Y
    val dYi: DenseMatrix[Double] = -4.0 :* (q * Y)
    //logInfo(s"dY before: $dY")
    //logInfo(s"dYi: $dYi")
    data.map(_._1).zipWithIndex.foreach{
      case (i, idx) => dY(i, ::) := dYi(idx, ::)
    }
    //logInfo(s"dY after: $dY")

    loss
  }
}
