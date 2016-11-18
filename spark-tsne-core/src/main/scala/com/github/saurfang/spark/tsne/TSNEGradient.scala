package com.github.saurfang.spark.tsne

import breeze.linalg._
import breeze.numerics._
import com.github.saurfang.spark.tsne.tree.SPTree
import org.slf4j.LoggerFactory

object TSNEGradient {
  def logger = LoggerFactory.getLogger(TSNEGradient.getClass)

  /**
    * Compute the numerator from the matrix Y
    *
    * @param idx the index in the matrix to use.
    * @param Y the matrix to analyze
    * @return the numerator
    */
  def computeNumerator(Y: DenseMatrix[Double], idx: Int *): DenseMatrix[Double] = {
    // Y_sum = ||Y_i||^2
    val sumY = sum(pow(Y, 2).apply(*, ::)) // n * 1
    val subY = Y(idx, ::).toDenseMatrix // k * 1
    val y1: DenseMatrix[Double] = Y * (-2.0 :* subY.t) // n * k
    val num: DenseMatrix[Double] = (y1(::, *) + sumY).t // k * n
    num := 1.0 :/ (1.0 :+ (num(::, *) + sumY(idx).toDenseVector)) // k * n

    idx.indices.foreach(i => num.update(i, idx(i), 0.0)) // num(i, i) = 0

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
    // q = (1 + ||Y_i - Y_j||^2)^-1 / sum(1 + ||Y_k - Y_l||^2)^-1
    val q: DenseMatrix[Double] = num / totalNum
    q.foreachPair{case ((i, j), v) => q.update(i, j, math.max(v, 1e-12))}

    // q = q - p
    val loss = data.zipWithIndex.flatMap {
      case ((_, itr), i) =>
        itr.map{
          case (j, p) =>
            val exaggeratedP = if(exaggeration) p * 4 else p
            val qij = q(i, j)
            val l = exaggeratedP * math.log(exaggeratedP / qij)
            q.update(i, j,  qij - exaggeratedP)
            if(l.isNaN) 0.0 else l
        }
    }.sum

    // l = [ (p_ij - q_ij) * (1 + ||Y_i - Y_j||^2)^-1 ]
    q :*= -num
    // l_sum = [0 0 ... sum(l) ... 0]
    sum(q(*, ::)).foreachPair{ case (i, v) => q.update(i, data(i)._1, q(i, data(i)._1) - v) }

    // dY_i = -4 * (l - l_sum) * Y
    val dYi: DenseMatrix[Double] = -4.0 :* (q * Y)
    data.map(_._1).zipWithIndex.foreach{
      case (i, idx) => dY(i, ::) := dYi(idx, ::)
    }

    loss
  }

  /** BH Tree related functions **/

  /**
   *
   * @param data array of (row_id, Seq(col_id), Vector(P_ij))
   * @param Y matrix
   * @param posF positive forces
   */
  def computeEdgeForces(data: Array[(Int, Seq[Int], DenseVector[Double])],
              Y: DenseMatrix[Double],
              posF: DenseMatrix[Double]): Unit = {
    data.foreach {
      case (i, cols, vec) =>
        // k x D - 1 x D  => k x D
        val diff = Y(cols, ::).toDenseMatrix.apply(*, ::) - Y(i, ::).t
        // k x D => k x 1
        val qZ = 1.0 :+ sum(pow(diff, 2).apply(*, ::))
        posF(i, ::) := (vec :/ qZ).t * (-diff)
    }
  }

  def computeNonEdgeForces(tree: SPTree,
                           Y: DenseMatrix[Double],
                           theta: Double,
                           negF: DenseMatrix[Double],
                           idx: Int *): Double = {
    idx.foldLeft(0.0)((acc, i) => acc + computeNonEdgeForce(tree, Y(i, ::).t, theta, negF, i))
  }

  /**
   * Calcualte negative forces using BH approximation
   *
   * @param tree SPTree used for approximation
   * @param y y_i
   * @param theta threshold for correctness / speed
   * @param negF negative forces
   * @param i row
   * @return sum of Q
   */
  private def computeNonEdgeForce(tree: SPTree,
                                  y: DenseVector[Double],
                                  theta: Double,
                                  negF: DenseMatrix[Double],
                                  i: Int): Double = {
    import tree._
    if(getCount == 0 || (isLeaf && center.equals(y))) {
      0.0
    } else {
      val diff = y - center
      val diffSq = sum(pow(diff, 2))
      if(isLeaf || radiusSq / diffSq < theta) {
        val qZ = 1 / (1 + diffSq)
        val nqZ = getCount * qZ
        negF(i, ::) :+= (nqZ * qZ * diff).t
        nqZ
      } else {
        children.foldLeft(0.0)((acc, child) => acc + computeNonEdgeForce(child, y, theta, negF, i))
      }
    }
  }

  def computeLoss(data: Array[(Int, Seq[Int], DenseVector[Double])],
                  Y: DenseMatrix[Double],
                  sumQ: Double): Double = {
    data.foldLeft(0.0){
      case (acc, (i, cols, vec)) =>
        val diff = Y(cols, ::).toDenseMatrix.apply(*, ::) - Y(i, ::).t
        val diffSq =  sum(pow(diff, 2).apply(*, ::))
        val Q = (1.0 :/ (1.0 :+ diffSq)) :/ sumQ
        sum(vec :* breeze.numerics.log(max(vec, 1e-12) :/ max(Q, 1e-12)))
    }
  }
}
