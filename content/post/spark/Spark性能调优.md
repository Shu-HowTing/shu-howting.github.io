+++
title = "Spark调优实战"
date = "2021-09-01"
description = "Spark调优实战"
tags = [
  "spark"
]
categories = [
  "spark"
]
draft = false
math = true
mathjax = true
+++

## Spark性能调优：合理设置并行度

### 1. Spark的并行度指的是什么？

并行度其实就是指的是spark作业中,各个stage的同时运行的task的数量,也就代表了spark作业在各个阶段stage的并行度！

$$
并行度 = executor\\_number * executor\\_cores
$$

**理解：**
sparkApplication的划分：
    $job --> stage --> task$
一般每个task一次处理一个分区。

**可以将task理解为比赛中的跑道：每轮比赛中，每个跑道上都会有一位运动员(分区，即处理的数据)，并行度就是跑道的数量，一轮比赛就可以理解为一个stage。**

### 2.如果不调节并行度，导致并行度过低会怎么样？
假设现在已经在spark-submit脚本里面，给我们的spark作业分配了足够多的资源，比如有50个 executor，每个executor 有10G内存，每个 executor 有3个cpu core，基本已经达到了集群或者yarn队列的资源上限。

如果 task 没有设置，或者设置的很少，比如就设置了100个 task。现在50个 executor，每个executor 有3个cpu core，也就是说，你的Application任何一个 stage 运行的时候都有总数在150个 cpu core，可以并行运行。但是你现在只有100个task，平均分配一下，每个executor 分配到2个task，那么同时在运行的task只有100个，每个executor只会并行运行2个task，每个executor剩下的一个 cpu core 就浪费掉了。

你的资源虽然分配足够了，但是问题是，并行度没有与资源相匹配，导致你分配下去的资源都浪费掉了。

合理的并行度的设置，应该是要设置的足够大，大到可以完全合理的利用你的集群资源。比如上面的例子，总共集群有150个cpu core，可以并行运行150个task。那么就应该将你的Application 的并行度至少设置成150才能完全有效的利用你的集群资源，让150个task并行执行，而且task增加到150个以后，既可以同时并行运行，还可以让每个task要处理的数据量变少。比如总共150G的数据要处理，如果是100个task，每个task计算1.5G的数据，现在增加到150个task可以并行运行，而且每个task主要处理1G的数据就可以。

很简单的道理，只要合理设置并行度，就可以完全充分利用你的集群计算资源，并且减少每个task要处理的数据量，最终，就是提升你的整个Spark作业的性能和运行速度。


### 3. 如何去提高并行度？
(1) task数量，至少设置成与spark Application 的总cpu core 数量相同（最理性情况，150个core，分配150task，一起运行，差不多同一时间运行完毕）

官方推荐，task数量，设置成spark Application 总cpu core数量的**2~3**倍 ，比如150个cpu core ，基本设置 task数量为 300~ 500， 与理性情况不同的，有些task 会运行快一点，比如50s 就完了，有些task 可能会慢一点，要一分半才运行完，所以如果你的task数量，刚好设置的跟cpu core 数量相同，可能会导致资源的浪费，因为 比如150task ，10个先运行完了，剩余140个还在运行，但是这个时候，就有10个cpu core空闲出来了，导致浪费。如果设置2~3倍，那么一个task运行完以后，另外一个task马上补上来，尽量让cpu core不要空闲。同时尽量提升spark运行效率和速度。提升性能。

(2) 如何设置一个Spark Application的并行度？
```
对于RDD来说：
可以通过设置spark.default.parallelism 参数来决定shuffle操作之后的partition数目，默认是没有值的，如果设置了值,比如说100，是在shuffle的过程才会起作用
new SparkConf().set(“spark.default.parallelism”, “500”)

对于sparksql来说：
可以通过设置spark.sql.shuffle.partitions参数，默认值为200;
```

(3) repartition算子
repartiton算子通过传入想要的分区数目，来改变分区数。注意：repartion是一个shuffle算子

(4) 在算子中加入指定的参数，来指定分区数目











