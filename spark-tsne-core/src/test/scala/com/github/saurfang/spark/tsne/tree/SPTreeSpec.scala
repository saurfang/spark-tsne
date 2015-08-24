package com.github.saurfang.spark.tsne.tree

import breeze.linalg._
import org.scalatest.{FunSpec, Matchers}

class SPTreeSpec extends FunSpec with Matchers {

  describe("SPTree") {
    describe("with 2 dimensions (quadtree)") {
      val tree = new SPTree(2, DenseVector(0.0, 0.0), DenseVector(2.0, 4.0))
      import tree._
      it("should have 4 children") {
        children.length shouldBe 4
      }
      it("each child should have correct width") {
        val width = DenseVector(1.0, 2.0)
        children.foreach(x => x.width shouldBe width)
      }
      it("children should have correct corner") {
        children.map(_.corner) shouldBe Array(
          DenseVector(0.0, 0.0),
          DenseVector(0.0, 2.0),
          DenseVector(1.0, 0.0),
          DenseVector(1.0, 2.0)
        )
      }
      it("getCell should return correct cell") {
        getCell(DenseVector(1.0, 1.0)).corner shouldBe DenseVector(0.0, 0.0)
        getCell(DenseVector(1.5, 1.5)).corner shouldBe DenseVector(1.0, 0.0)
        getCell(DenseVector(2.0, 2.0)).corner shouldBe DenseVector(1.0, 0.0)
        getCell(DenseVector(2.0, 2.5)).corner shouldBe DenseVector(1.0, 2.0)
      }
      it("should be able to be constructed from DenseMatrix") {
        val data = Array(
          1.0, 1.0, 1.0, 2.0, 1.1, 1.11, 1.11, 1,
          3.0, 1.0, 2.0, 2.0, 1.1, 1.11, 1.11, 1
        )
        val matrix = DenseMatrix.create[Double](data.length / 2, 2, data)
        val tree = SPTree(matrix)

        tree.getCount shouldBe matrix.rows
        tree.children.map(_.getCount).sum shouldBe matrix.rows
        tree.center shouldBe DenseVector(data.grouped(matrix.rows).map(x => x.sum / x.length).toArray)
        verifyCorrectness(tree)
      }
    }
  }

  def verifyCorrectness(tree: SPTree): Unit = {
    if(tree.getCount <= 1) tree.isLeaf shouldBe true
    if(tree.getCount > 0) tree.center shouldBe (tree.totalMass / tree.getCount.toDouble)
    if(tree.isLeaf) {
      tree.children.foreach(_.isLeaf shouldBe true)
      tree.children.foreach(_.getCount shouldBe 0)
    } else {
      tree.children.map(_.getCount).sum shouldBe tree.getCount
      val totalMassTally = tree.children.foldLeft(DenseVector.zeros[Double](tree.dimension))((acc, t) => acc + t.totalMass)
      (0 until tree.dimension).foreach(i => totalMassTally(i) shouldBe (tree.totalMass(i) +- 1e-5))
      tree.children.foreach(verifyCorrectness)
    }
  }
}
