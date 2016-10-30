import sbt._
import Keys._

object Dependencies {
  val Versions = Seq(
    crossScalaVersions := Seq("2.11.8", "2.10.5"),
    scalaVersion := crossScalaVersions.value.head
  )

  object Compile {
    val spark = "org.apache.spark" %% "spark-mllib" % "2.0.1" % "provided"
    val breeze_natives = "org.scalanlp" %% "breeze-natives" % "0.11.2" % "provided"
    val logging = "com.typesafe.scala-logging" %% "scala-logging" % "3.4.0"

    object Test {
      val scalatest = "org.scalatest" %% "scalatest" % "3.0.0" % "test"
    }
  }

  import Compile._
  val l = libraryDependencies

  val core = l ++= Seq(spark, breeze_natives, logging, Test.scalatest)
}
