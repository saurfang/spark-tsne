import sbtsparksubmit.SparkSubmitPlugin.autoImport._

object SparkSubmit {
  lazy val settings =
    SparkSubmitSetting("sparkMNIST",
      Seq(
        "--master", "local[*]",
        "--class", "com.github.saurfang.spark.tsne.examples.MNIST"
      )
    )
}
