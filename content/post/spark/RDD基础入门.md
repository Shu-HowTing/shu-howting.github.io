+++
title = "Spark RDD入门"
date = "2019-09-05"
description = "Spark RDD入门"
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

### RDD简介

RDD--弹性分布式数据集（Resilient Distributed Dataset）是spark的核心概念。RDD其实就是分布式的元素集合。在Spark中，对数据的所有操作不外乎创建RDD，转化已有的RDD以及调用RDD操作进行求值。而在这一切的背后，spark会自动讲RDD中的数据分发到集群上，并将操作并行化执行。

### RDD基础
RDD是一个不可变的分布式对象集合.每个RDD被分为多个分区，这些分区运行在集群中的不同节点上。RDD可以包含Python、Java、Scala中任意类型的对象。

**每个spark程序无外乎都是下面的流程:**

1. 从外部数据创建输入RDD 

2. 使用诸如filter()这样的操作对RDD进行转化，定义新的RDD 

3. 告诉spark对需要被重用的中间RDD执行persisit()操作

4. 使用行动操作(count(),first())触发一次并行计算，spark并不会立马执行，而是优化后再执行

#### 1. 创建RDD
**Spark提供了两种方法创建RDD的方法：**

- 读取外部数据集
- 在驱动器程序中对一个集合进行并行化

```python
#parallelize方法将集合转化为rdd
lines = sc.parallelize(["pandas", "i like pandas"])

#textFile方法
lines=sc.textFile("README.md")
```

#### 2. 创建RDD
**spark支持两种操作：转化操作，行动操作**

##### 转化操作 

转化操作执行时返回新的RDD的操作，转化出来的RDD是惰性求值的，只有在行动中用到这些RDD时才会被计算，许多转化操作只会操作RDD中的一个元素，并不是所有的转化操作都是这样

比如提取日志文件的错误信息


```python
inputRDD=sc.testFile("log.txt")
errorsRDD=inputRDD.filter(lambda x:"error" in x)
warningsRDD = inputRDD.filter(lambda x: "warning" in x)
# 将errorsRDD和warningsRDD合并成一个RDD
badLinesRDD = errorsRDD.union(warningsRDD)
```

通过转化操作，从已有的RDD中派生新的RDD，spark会使用谱系图记录这些RDD的依赖关系，spark在需要用这些信息的时候按需计算每个RDD，也可以依靠谱系图在丢失数据的情况下恢复丢失的数据

##### 行动操作 
行动操作需要实际的输出，它会强制执行哪些求值必须用到的RDD转化操作 

示例：对badLinesRDD进行计数操作，并且打印前十条记录


```python
print "Input had " + badLinesRDD.count() + " concerning lines"
# take(num) 从RDD中取出num个元素
for line in badLinesRDD.take(10):
    print line
```

这里使用了take()取出少量的数据集，也可以使用collect()函数获取整个RDD中的数据，但是使用collect需要注意内存是否够用。如果数据集特别大的时候，我们需要把数据写到诸如HDFS之类的分布式存储系统，当调用一个新的行动操作的时候整个RDD会从头计算，我们要将中间结果持久化

##### 惰性求值
RDD的转化操作都是多心求值的，这意味着在被调用行动操作之前Spark不会开始计算。

惰性求值意味着我们对RDD调用转化操作（例如map()）时，操作不会立即执行，相反，Spark会在内部记录下所要求执行的操作的相关信息。**我们不应该把RDD看作放着特定数据的数据集，而最好把每个RDD看作我们通过转化操作构建出来的、记录如何计算数据的指令列表**。把数据读到RDD的操作也是惰性的，因此，当我们调用sc.textFile()时，数据并没有读取进来，而是在必要时才会读取。

#### 3. 创建RDD

```python
#使用lambda方法传递
word = rdd.filter(lambda s: "error" in s)

#定义一个函数然后传递
def containsError(s):
    return "error" in s
word = rdd.filter(containsError)
```
传递函数时要小心，python会在不经意间把函数所在的对象也序列化传出，有时如果传递的类里包含了python不知道如何序列化输出的对象，也可能导致程序失败。

如下是一个错误的函数传递示例；

```python
class SearchFunctions(object):
    def __init__(self, query):
        self.query = query
    def isMatch(self, s):
        return self.query in s
    def getMatchesFunctionReference(self, rdd):
        # 问题：在"self.isMatch"中引用了整个self
        return rdd.filter(self.isMatch)
    def getMatchesMemberReference(self, rdd):
        # 问题：在"self.query"中引用了整个self
        return rdd.filter(lambda x: self.query in x)
```
正确做法:

```python
class WordFunctions(object):
    def getMatchesNoReference(self, rdd):
        # 安全：只把需要的字段提取到局部变量中
        query = self.query
        return rdd.filter(lambda x: query in x)
```
#### 4、RDD操作
常见的转化操作

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/rdd1.png)

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/rdd2.png)

常见的行动操作

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/rdd3.png)

#### 5、持久化(缓存)
如前所述， Spark 
RDD是惰性求值的，而有时我们希望能多次使用同一个 RDD。如果简单
地对 RDD 调用行动操作， Spark 每次都会重算 RDD 以及它的所有依赖。这在迭代算法中
消耗格外大，因为迭代算法常常会多次使用同一组数据。

如下就是先对 RDD 作一次计数、再把该 RDD 输出的一个小例子。

```scala
val result = input.map(x => x*x)
println(result.count())
println(result.collect().mkString(","))
```
为了避免多次计算同一个 RDD，可以让 Spark 对数据进行持久化。当我们让 Spark 持久化
存储一个 RDD 时，计算出 RDD 的节点会分别保存它们所求出的分区数据。如果一个有持
久化数据的节点发生故障， Spark 会在需要用到缓存的数据时重算丢失的数据分区。如果
希望节点故障的情况不会拖累我们的执行速度，也可以把数据备份到多个节点上。

出于不同的目的，我们可以为 RDD 选择不同的持久化级别（如表 3-6 所示）。在 Scala和 Java 中，默认情况下 persist() 会把数据以序列化的形式缓存在 JVM 的堆空
间中。在 Python 中，我们会始终序列化要持久化存储的数据，所以持久化级别默认值就是
以序列化后的对象存储在 JVM 堆空间中。当我们把数据写到磁盘或者堆外存储上时，也
总是使用序列化后的数据。

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/rdd4.png)

```Scala
val result = input.map(x => x * x)
result.persist(StorageLevel.DISK_ONLY)
println(result.count())
println(result.collect().mkString(","))
```
如果要缓存的数据太多， 内存中放不下， Spark 会自动利用最近最少使用（ LRU）的缓存
策略把最老的分区从内存中移除。 对于仅把数据存放在内存中的缓存级别，下一次要用到
已经被移除的分区时， 这些分区就需要重新计算。但是对于使用内存与磁盘的缓存级别的
分区来说，被移除的分区都会写入磁盘。不论哪一种情况，都不必担心你的作业因为缓存
了太多数据而被打断。 不过，缓存不必要的数据会导致有用的数据被移出内存，带来更多
重算的时间开销