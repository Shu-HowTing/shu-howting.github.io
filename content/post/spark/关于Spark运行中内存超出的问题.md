+++
title = "Spark运行内存超出"
date = "2020-03-01"
description = "Spark运行内存超出"
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

## Container killed by YARN for exceeding memory limits？

`运行spark脚本时，经常会碰到Container killed by YARN for exceeding memory limits的错误，导致程序运行失败。`

`这个的意思是指executor的外堆内存超出了。默认情况下，这个值被设置为executor_memory的10%或者384M，以较大者为准，即max(executor_memory*.1, 384M).` 

### 解决办法
- 提高内存开销
- 减少执行程序内核的数量
- 增加分区数量
- 提高驱动程序和执行程序内存


#### 提高内存开销
即直接指定堆外内存的大小 
```
spark.conf.set("spark.yarn.executor.memoryOverhead", "4g")
```

#### 减少执行程序内核的数量

这可减少执行程序可以执行的最大任务数量，从而减少所需的内存量。


#### 增加分区数量
要增加分区数量，请为原始弹性分布式数据集增加 spark.default.parallelism 的值，或执行.repartition() 操作。增加分区数量可减少每个分区所需的内存量。

#### 提高驱动程序和执行程序内存
即通过增大executor.memory的值来增大堆外内存,但是可以看到，由于乘了10%，所以提升其实很有限。







