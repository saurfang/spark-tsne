#spark-tsne

[![Join the chat at https://gitter.im/saurfang/spark-tsne](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/saurfang/spark-tsne?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/erwinvaneijk/spark-tsne.svg?branch=master)](https://travis-ci.org/erwinvaneijk/spark-tsne)
Distributed [t-SNE](http://lvdmaaten.github.io/tsne/) with Apache Spark. WIP...

t-SNE is a dimension reduction technique that is particularly good for visualizing high
dimensional data. This is an attempt to implement this algorithm using Spark to leverage
distributed computing power.

The project is still in progress of replicating reference implementations from the original
papers. Spark specific optimizations will be the next goal once the correctness is verified.

Currently I'm showcasing this using the standard [MNIST](http://yann.lecun.com/exdb/mnist/)
handwriting recognition dataset. I have created a [WebGL player](https://saurfang.github.io/spark-tsne-demo/tsne-pixi.html)
(built using [pixi.js](https://github.com/pixijs/pixi.js)) to visualize the inner workings
as well as the final results of t-SNE. If a WebGL is unavailable for you, you may checkout
the [d3.js player](https://saurfang.github.io/spark-tsne-demo/tsne.html) instead.

![](data/mnist/tsne.gif)

## Credits

- [t-SNE Julia implementation](https://github.com/lejon/TSne.jl)
- [Barnes-Hut t-SNE](https://github.com/lvdmaaten/bhtsne/)
