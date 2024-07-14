+++
title = "RDD算子总结"
date = "2021-09-01"
description = "RDD算子总结"
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

## RDD算子总结

### 从功能上分：
**转换算子(transformer)：** lazy执行，生成新的rdd，只有在调用action算子时，才会真正的执行。
如：`$map 、flatmap、filter、   union、  join、  ruduceByKey、 cache$`

**行动算子(action)：** 触发任务执行，产生job，返回值不再是rdd。
如：`$count 、collect、top、  take、  reduce$`

### 从作用上分：
**通用的：** map、 flatMap、 distinct、 union

**作用于RDD[K,V]：** mapValues、 reduceByKey、 groupByKey、  sortByKey、 


### 转换算子是否有shuffle

**shuffle类:** reduceByKey、 groupByKey、 groupBy、 join、 distinct、 repartition

**非shuffle类:** map、 filter、 union、flatMap、  coalesce

[Spark算子使用案例总结](http://homepage.cs.latrobe.edu.au/zhe/ZhenHeSparkRDDAPIExamples.html)