+++
title = "repartition和coalesce区别"
date = "2019-10-15"
description = "repartition和coalesce区别"
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

### 简介
$repartition(numPartitions:Int)$ 和 $coalesce(numPartitions:Int，shuffle:Boolean=false)$

- 作用：对RDD的分区进行**重新划分**，repartition内部调用了coalesce，参数`$shuffle=true$`
### 分析
#### 例：RDD有N个分区，需要重新划分成M个分区

- N小于M 

一般情况下N个分区有数据分布不均匀的状况，利用HashPartitioner函数将数据重新分区为M个，这时需要将shuffle设置为true。

- N大于M且和M相差不多

假如N是1000，M是100。那么就可以将N个分区中的若干个分区合并成一个新的分区，最终合并为M个分区，这时可以将shuff设置为false，在shuffl为false的情况下，如果M>N时，coalesce为无效的，不进行shuffle过程，父RDD和子RDD之间是窄依赖关系。

- N大于M且和M相差悬殊

这时如果将shuffle设置为false，父子RDD是窄依赖关系，他们在同一个Stage中，就可能造成Spark程序的并行度不够，从而影响性能，如果在M为1的时候，为了使coalesce之前的操作有更好的并行度，可以讲shuffle设置为true。


### 总结：
返回一个减少到numPartitions个分区的新RDD，这会导致窄依赖
例如：你将1000个分区转换成100个分区，这个过程不会发生shuffle，相反如果10个分区转换成100个分区将会发生shuffle。
然而如果你想大幅度合并分区，例如所有partition合并成一个分区，这会导致计算在少数几个集群节点上进行（言外之意：并行度不够）。
'为了避免这种情况，你可以将第二个shuffle参数传递一个true，这样会在重新分区过程中多一步shuffle，这意味着上游的分区可以并行运行。

总之：
如果shuff为false时，如果传入的参数大于现有的分区数目，RDD的分区数不变，也就是说不经过shuffle，是无法将RDD的partition数变多的。


### 进一步理解
```
N大于M且和M相差悬殊
将shuffle设置为false，父子RDD是窄依赖关系，他们在同一个Stage中，就可能造成Spark程序的并行度不够，从而影响性能，如
果在M为1的时候，为了使coalesce之前的操作有更好的并行度，可以将shuffle设置为true
```

**每个Stage里面的Task的数量是由该Stage中最后一个RDD的Partition的数量所决定的！！**
```
和repartition有所区别的是，coalesce并不是一个shuffle算子。也就说coalesce不会触发shuffle操作，它是包含在当前的stage
中的。由于，每个Stage里面的Task的数量是由该Stage中最后一个RDD的Partition的数量所决定的。就会引起这样一种现象：由于
coalesce算子的存在，必然导致运算后的partition数目的减少。也就是说当前的stage的并行的task数目(并行度)会降低。每个ta
sk计算的数据(分区)会加大。可以将shuffle设置为true来触发shuffle，从而不会降低当前stage的task数量(并行度)。
```

![image](https://tva1.sinaimg.cn/large/006tNbRwgy1gajo18ko6wj30uf0chabb.jpg)

> 由图可见，在coalesce操作之前，每个rdd有四个partition，如果没有soalesce操作,当前stage的并行度为4.但是由于coalesce
操作的存在，导致分区数变为2。所以整个stage的并行度为2，在实际运行时，excutor只会为当前的stage启动2个核。也就是说输入的4个partition会被分布到两个task上运行，每个task上分到两个partion。每个task运算的数据量变大，运行速度就会被拖慢。














