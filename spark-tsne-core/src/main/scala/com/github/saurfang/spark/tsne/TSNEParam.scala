package com.github.saurfang.spark.tsne

case class TSNEParam(
                      early_exaggeration: Int = 100,
                      exaggeration_factor: Double = 4.0,
                      t_momentum: Int = 25,
                      initial_momentum: Double = 0.5,
                      final_momentum: Double = 0.8,
                      eta: Double = 500.0,
                      min_gain: Double = 0.01
                      )
