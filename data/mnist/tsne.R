library(dplyr)
library(ggplot2)
library(animation)

resultFiles <- list.files("~/GitHub/spark-tsne/.tmp/MNIST/", "result", full.names = TRUE)
results <- lapply(resultFiles, function(file) { read.csv(file, FALSE) })

plotResult <- function(i) {
  ggplot(results[[i]]) +
    aes(V2, V3, color = as.factor(V1)) +
    geom_point()
}

traceAnimate <- function() {
  lapply(seq(1, length(results), 1), function(i) {
    plotResult(i)
  })
}

saveGIF(traceAnimate, interval = .2, movie.name = "tsne.gif")

