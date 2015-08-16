library(dplyr)
library(ggplot2)
library(animation)

resultFiles <- list.files("~/GitHub/spark-tsne/.tmp/MNIST/", "result", full.names = TRUE)
results <- lapply(resultFiles, function(file) { read.csv(file, FALSE) })

computeLimit <- function(f, cumf) {
  cumf(lapply(results, f))
}

xmax <- computeLimit(. %>% {max(abs(.$V2))}, cummax)
ymax <- computeLimit(. %>% {max(abs(.$V3))}, cummax)

plotResult <- function(i) {
  ggplot(results[[i]]) +
    aes(V2, V3, color = as.factor(V1), label = V1) +
    #geom_point() +
    geom_text() +
    xlim(-xmax[i], xmax[i]) +
    ylim(-ymax[i], ymax[i])
}

traceAnimate <- function() {
  lapply(seq(1, length(results), 1), function(i) {
    print(plotResult(i))
  })
}

file.remove("tsne.gif")
saveGIF(traceAnimate(), interval = 0.05, movie.name = "tsne.gif", loop = 1)

