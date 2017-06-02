package com.github.saurfang.spark.tsne.tree

import breeze.linalg._
import breeze.numerics._

import scala.annotation.tailrec


class SPTree private[tree](val dimension: Int,
              val corner: DenseVector[Double],
              val width: DenseVector[Double]) extends Serializable {
  private[this] val childWidth: DenseVector[Double] = width :/ 2.0
  lazy val radiusSq: Double = sum(pow(width, 2))
  private[tree] val totalMass: DenseVector[Double] = DenseVector.zeros(dimension)
  private var count: Int = 0
  private var leaf: Boolean = true
  val center: DenseVector[Double] = DenseVector.zeros(dimension)

  lazy val children: Array[SPTree] = {
    (0 until pow(2, dimension)).toArray.map {
      i =>
        val bits = DenseVector(s"%0${dimension}d".format(i.toBinaryString.toInt).toArray.map(_.toDouble - '0'.toDouble))
        val childCorner: DenseVector[Double] = corner + (bits :* childWidth)
        new SPTree(dimension, childCorner, childWidth)
    }
  }

  final def insert(vector: DenseVector[Double], finalize: Boolean = false): SPTree = {
    totalMass += vector
    count += 1

    if(leaf) {
      if(count == 1) { // first to leaf
        center := vector
      } else if(!vector.equals(center)) {
        (1 until count).foreach(_ => getCell(center).insert(center, finalize)) //subdivide
        leaf = false
      }
    }

    if(finalize) computeCenter(false)

    if(leaf) this else getCell(vector).insert(vector, finalize)
  }

  def computeCenter(recursive: Boolean = true): Unit = {
    if(count > 0) {
      center := totalMass / count.toDouble
      if(recursive) children.foreach(_.computeCenter())
    }
  }

  def getCell(vector: DenseVector[Double]): SPTree = {
    val idx = ((vector - corner) :/ childWidth).data
    children(idx.foldLeft(0)((acc, i) => acc * 2 + min(max(i.ceil.toInt - 1, 0), 1)))
  }

  def getCount: Int = count

  def isLeaf: Boolean = leaf
}

object SPTree {
  def apply(Y: DenseMatrix[Double]): SPTree = {
    val d = Y.cols
    val minMaxs = minMax(Y(::, *)).t
    val mins = minMaxs.mapValues(_._1)
    val maxs = minMaxs.mapValues(_._2)

    val tree = new SPTree(Y.cols, mins, maxs - mins)

    // insert points but wait till end to compute all centers
    //Y(*, ::).foreach(tree.insert(_, finalize = false))
    (0 until Y.rows).foreach(i => tree.insert(Y(i, ::).t, finalize = false))
    // compute all center of mass
    tree.computeCenter()

    tree
  }
}