library(dplyr)
library(ggplot2)
library(animation)
library(jsonlite)

resultFiles <- list.files("~/GitHub/spark-tsne/.tmp/MNIST/", "result", full.names = TRUE)
results <- lapply(resultFiles, function(file) { read.csv(file, FALSE) })

#### save results as json for viewer ####
resultsByObs <- lapply(1:nrow(results[[1]]), function(i) {
  list(
    key = unbox(i),
    label = unbox(results[[1]]$V1[i]),
    x = data.frame(i = 1:length(results), x = sapply(results, . %>% {.$V2[i]} )),
    y = data.frame(i = 1:length(results), y = sapply(results, . %>% {.$V3[i]} ))
  )
})
write(toJSON(resultsByObs, "values"), "mnist.json")

#### save plot as animated gif ####
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

traceAnimate <- function(length = length(results), step = 1) {
  lapply(seq(1, length, step), function(i) {
    print(plotResult(i))
  })
}

file.remove("tsne.gif")
saveGIF(traceAnimate(step = 5), interval = 0.05, movie.name = "tsne.gif", loop = 1)
