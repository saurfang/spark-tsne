package com.github.saurfang.spark.tsne

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}

/**
  * This test demonstrates the bug introduced when upgrading the codebase to spark 2.1.
  *
  * For completeness and to check regressions, it's now added to the codebase.
  *
  * @author erwin.vaneijk@gmail.com
  */
class BugDemonstrationTest extends FunSuite with Matchers with BeforeAndAfterAll {
  private var sparkSession : SparkSession = _
  override def beforeAll(): Unit = {
    super.beforeAll()
    sparkSession = SparkSession.builder().appName("BugTests").master("local[2]").getOrCreate()
  }

  override def afterAll(): Unit = {
    super.afterAll()
    sparkSession.stop()
  }

  test("This demonstrates a bug was fixed in tsne-spark 2.1") {
    val sc = sparkSession.sparkContext

    val observations = sc.parallelize(
      Seq(
        Vectors.dense(1.0, 10.0, 100.0),
        Vectors.dense(2.0, 20.0, 200.0),
        Vectors.dense(3.0, 30.0, 300.0)
      )
    )

    // Compute column summary statistics.
    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
    val expectedMean = Vectors.dense(2.0,20.0,200.0)
    val resultMean = summary.mean
    assertEqualEnough(resultMean, expectedMean)
    val expectedVariance = Vectors.dense(1.0,100.0,10000.0)
    assertEqualEnough(summary.variance, expectedVariance)
    val expectedNumNonZeros = Vectors.dense(3.0, 3.0, 3.0)
    assertEqualEnough(summary.numNonzeros, expectedNumNonZeros)
  }

  private def assertEqualEnough(sample: Vector, expected: Vector): Unit = {
    expected.toArray.zipWithIndex.foreach{ case(d: Double, i: Int) =>
      sample(i) should be (d +- 1E-12)
    }
  }
}
