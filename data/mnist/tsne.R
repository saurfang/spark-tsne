library(dplyr)
library(ggplot2)
library(animation)
library(jsonlite)

resultFiles <- list.files("~/GitHub/spark-tsne/.tmp/MNIST/", "result", full.names = TRUE)
results <- lapply(resultFiles, function(file) { read.csv(file, FALSE) })
resultsCombined <- lapply(1:length(results), function(i) {
  result <- results[[i]]
  names(result)  <- c("label", "x", "y")
  mutate(result, i = i, key = row_number())
}) %>%
  rbind_all()

#### save results as json for viewer ####
iterations <- c(1:99, seq(100, length(results), 5)) # assume 100 early exaggeration here
resultsByObs <- filter(resultsCombined, i %in% iterations) %>%
  group_by(key) %>%
#   do({
#     list(key = unbox(.$key[1]), label = unbox(.$label[1]),
#          # assume order will preserve
#          pos = select(., x, y)) %>%
#     data_frame
#   })
  do(key = unbox(.$key[1]),
     label = unbox(.$label[1]),
     pos = select(., x, y))
write(toJSON(list(iterations = iterations, data = resultsByObs)), "mnist.json")

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

traceAnimate <- function(n = length(results), step = 1) {
  lapply(seq(1, n, step), function(i) {
    print(plotResult(i))
  })
}

file.remove("tsne.gif")
saveGIF(traceAnimate(step = 5), interval = 0.05, movie.name = "tsne.gif", loop = 1)
