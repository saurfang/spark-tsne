import sbtsparksubmit.SparkSubmitPlugin.autoImport._

object SparkSubmit {
  lazy val settings =
    SparkSubmitSetting("sparkMNIST",
      Seq(
        "--master", "local[2]",
        "--class", "com.github.saurfang.spark.tsne.examples.MNIST"
      )
    )
}
