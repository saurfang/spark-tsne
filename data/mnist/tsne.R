library(dplyr)
library(ggplot2)
library(animation)

resultFiles <- list.files("~/GitHub/spark-tsne/.tmp/MNIST/", "result", full.names = TRUE)
results <- lapply(resultFiles, function(file) { read.csv(file, FALSE) })

xlimit <- sapply(results, . %>% { .$V2 }) %>% {c(min(.), max(.))}
ylimit <- sapply(results, . %>% { .$V3 }) %>% {c(min(.), max(.))}

plotResult <- function(i) {
  ggplot(results[[i]]) +
    aes(V2, V3, color = as.factor(V1)) +
    geom_point() +
    xlim(xlimit) +
    ylim(ylimit)
}

traceAnimate <- function() {
  lapply(seq(1, length(results), 1), function(i) {
    print(plotResult(i))
  })
}

saveGIF(traceAnimate(), interval = 0.05, movie.name = "tsne.gif", loop = 1)

