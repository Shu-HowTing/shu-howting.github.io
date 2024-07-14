+++
title = "Spark2.0特性"
date = "2021-09-01"
description = "Spark2.0特性"
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


# Spark2.0

Spark直接从1.6跨入2.0版本，带来一些新的特性。最大的变化便是SparkSession整合了各种环境。

Spark2.0中引入了`$SparkSession$`的概念，它为用户提供了一个统一的切入点来使用Spark的各项功能，用户不但可以使用DataFrame和Dataset的各种API，学习Spark的难度也会大大降低。


## SparkSession
在Spark的早期版本，SparkContext是进入Spark的切入点。我们都知道RDD是Spark中重要的API，然而它的创建和操作得使用sparkContext提供的API；对于RDD之外的其他东西，我们需要使用其他的Context。比如对于流处理来说，我们得使用StreamingContext；对于SQL得使用sqlContext；而对于hive得使用HiveContext。然而DataSet和Dataframe提供的API逐渐称为新的标准API，我们需要一个切入点来构建它们，所以在 Spark 2.0中我们引入了一个新的切入点(entry point)：SparkSession

　SparkSession实质上是SQLContext和HiveContext的组合（未来可能还会加上StreamingContext），所以在SQLContext和HiveContext上可用的API在SparkSession上同样是可以使用的。SparkSession内部封装了sparkContext，所以计算实际上是由sparkContext完成的。

**之前的写法：**
```py
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

conf = SparkConf().setMaster("local[*]").setAppName("PySparkShell")
sc = SparkContext(conf=conf)
sqlContest = SQLContext(sc)
spark = SQLContext(sc)
spark.sql(select **)
···
```
**现在的写法**
```py
from pyspark.sql import SparkSession
spark = SparkSession 
        .builder 
        .appName("Python Spark SQL basic example")
        .config("spark.some.config.option","some-value") 
        .enableHiveSupport()
        .getOrCreate()
df1 = spark.sql(select **)    

df2 = spark.read.csv('./python/test_support/sql/ages.csv')
 
 
# 通过spark创建sc
sc = spark.sparkContext
rdd1 = sc.parallelize([1,2,3,4,5])
```
#### 其中：
- 在pyspark sql中换行要  \
- .getOrCreate() 指的是如果当前存在一个SparkSession就直接获取，否则新建。
- .enableHiveSupport() 使我们可以从读取或写入数据到hive。
.enableHiveSupport 函数的调用使得SparkSession支持hive，类似于HiveContext