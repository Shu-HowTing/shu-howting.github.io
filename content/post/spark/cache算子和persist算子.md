+++
title = "cache和persist比较"
date = "2020-03-10"
description = "cache和persist比较"
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

## Spark中cache和persist的作用

Spark开发高性能的大数据计算作业并不是那么简单。如果没有对Spark作业进行合理的调优，Spark作业的执行速度可能会很慢，这样就完全体现不出Spark作为一种快速大数据计算引擎的优势来。因此，想要用好Spark，就必须对其进行合理的性能优化。

有一些代码开发基本的原则，避免创建重复的RDD，尽可能复用同一个RDD，如下，我们可以直接用一个RDD进行多种操作：
```scala
val rdd1 = sc.textFile("xxx")
rdd1.collect
rdd1.show
```

但是Spark中对于一个RDD执行多次算子的默认原理是这样的：每次你对一个RDD执行一个算子操作时，都会重新从源头处计算一遍，计算出那个RDD来，然后再对这个RDD执行你的算子操作。对于上面的代码，`sc.textFile("xxx")`会执行两次，这种方式的性能是很差的。

因此对于这种情况，我的建议是：对多次使用的RDD进行持久化。此时Spark就会根据你的持久化策略，将RDD中的数据保存到内存或者磁盘中。以后每次对这个RDD进行算子操作时，都会直接从内存或磁盘中提取持久化的RDD数据，然后执行算子，而不会从源头处重新计算一遍这个RDD，再执行算子操作。

### 持久化
如果要对一个RDD进行持久化，只要对这个RDD调用`cache()`和`persist()`即可。
```scala
val rdd1 = sc.textFile("xxx").cache
rdd1.collect
rdd1.show
```

1. cache()方法表示：使用非序列化的方式将RDD中的数据全部尝试持久化到内存中。

此时再对rdd1执行两次算子操作时，只有在第一次算子时，才会将这个rdd1从源头处计算一次。第二次执行算子时，就会直接从内存中提取数据进行计算，不会重复计算一个rdd。
```scala
/**
 * Persist this RDD with the default storage level (`MEMORY_ONLY`).
 */
def cache(): this.type = persist()
```

**通过源码可以看出cache()是persist()的简化方式，调用persist的无参版本，也就是调用persist(StorageLevel.MEMORY_ONLY)，cache只有一个默认的缓存级别MEMORY_ONLY，即将数据持久化到内存中**

2. persist()方法表示：手动选择持久化级别，并使用指定的方式进行持久化。

```scala
/**
 * Persist this RDD with the default storage level (`MEMORY_ONLY`).
*/
def persist(): this.type = persist(StorageLevel.MEMORY_ONLY)
```
默认缓存级别是`StorageLevel.MEMORY\_ONLY`.也就是说cache的默认级别就是`MEMORY\_ONLY`

#### DataFrame的cache和persist的区别
官网和上的教程说的都是RDD,但是没有讲df的缓存，通过源码发现df和rdd还是不太一样的:
```scala
/**
 * Persist this Dataset with the default storage level (`MEMORY_AND_DISK`).
 *
 * @group basic
 * @since 1.6.0
 */
def cache(): this.type = persist()

/**
 * Persist this Dataset with the default storage level (`MEMORY_AND_DISK`).
 *
 * @group basic
 * @since 1.6.0
 */
def persist(): this.type = {
  sparkSession.sharedState.cacheManager.cacheQuery(this)
  this
}


def persist(newLevel: StorageLevel): this.type = {
  sparkSession.sharedState.cacheManager.cacheQuery(this, None, newLevel)
  this
}

def cacheQuery(
    query: Dataset[_],
    tableName: Option[String] = None,
    storageLevel: StorageLevel = MEMORY_AND_DISK): Unit = writeLock
```
**可以到cache()依然调用的persist()，但是persist调用cacheQuery，而cacheQuery的默认存储级别为MEMORY_AND_DISK，这点和rdd是不一样的。**

## RDD的缓存级别
每个持久化的 RDD 可以使用不同的存储级别进行缓存，例如，持久化到磁盘、已序列化的 Java 对象形式持久化到内存（可以节省空间）、跨节点间复制、以 off-heap 的方式存储在 Tachyon。这些存储级别通过传递一个 StorageLevel 对象给 persist() 方法进行设置。
详细的存储级别介绍如下：

- MEMORY_ONLY : 将 RDD 以反序列化 Java 对象的形式存储在 JVM 中。如果内存空间不够，部分数据分区将不再缓存，在每次需要用到这些数据时重新进行计算。这是默认的级别。
- MEMORY_AND_DISK : 将 RDD 以反序列化 Java 对象的形式存储在 JVM 中。如果内存空间不够，将未缓存的数据分区存储到磁盘，在需要使用这些分区时从磁盘读取。
- MEMORY_ONLY_SER : 将 RDD 以序列化的 Java 对象的形式进行存储（每个分区为一个 byte 数组）。这种方式会比反序列化对象的方式节省很多空间，尤其是在使用 fast serializer时会节省更多的空间，但是在读取时会增加 CPU 的计算负担。
- MEMORY_AND_DISK_SER : 类似于 MEMORY_ONLY_SER ，但是溢出的分区会存储到磁盘，而不是在用到它们时重新计算。
- DISK_ONLY : 只在磁盘上缓存 RDD。
- MEMORY_ONLY_2，MEMORY_AND_DISK_2，等等 : 与上面的级别功能相同，只不过每个分区在集群中两个节点上建立副本。
- OFF_HEAP（实验中）: 类似于 MEMORY_ONLY_SER ，但是将数据存储在 off-heap memory，这需要启动 off-heap 内存。

注意，在 Python 中，缓存的对象总是使用 Pickle 进行序列化，所以在 Python 中不关心你选择的是哪一种序列化级别。python 中的存储级别包括 MEMORY_ONLY，MEMORY_ONLY_2，MEMORY_AND_DISK，MEMORY_AND_DISK_2，DISK_ONLY 和 DISK_ONLY_2 。

## 持久化策略的选择

- 默认情况下，性能最高的当然是MEMORY_ONLY，但前提是你的内存必须足够足够大，可以绰绰有余地存放下整个RDD的所有数据。因为不进行序列化与反序列化操作，就避免了这部分的性能开销；对这个RDD的后续算子操作，都是基于纯内存中的数据的操作，不需要从磁盘文件中读取数据，性能也很高；而且不需要复制一份数据副本，并远程传送到其他节点上。但是这里必须要注意的是，在实际的生产环境中，恐怕能够直接用这种策略的场景还是有限的，如果RDD中数据比较多时（比如几十亿），直接用这种持久化级别，会导致JVM的OOM内存溢出异常。

- 如果使用MEMORY_ONLY级别时发生了内存溢出，那么建议尝试使用MEMORY_ONLY_SER级别。该级别会将RDD数据序列化后再保存在内存中，此时每个partition仅仅是一个字节数组而已，大大减少了对象数量，并降低了内存占用。这种级别比MEMORY_ONLY多出来的性能开销，主要就是序列化与反序列化的开销。但是后续算子可以基于纯内存进行操作，因此性能总体还是比较高的。此外，可能发生的问题同上，如果RDD中的数据量过多的话，还是可能会导致OOM内存溢出的异常。

- 如果纯内存的级别都无法使用，那么建议使用MEMORY_AND_DISK_SER策略，而不是MEMORY_AND_DISK策略。因为既然到了这一步，就说明RDD的数据量很大，内存无法完全放下。序列化后的数据比较少，可以节省内存和磁盘的空间开销。同时该策略会优先尽量尝试将数据缓存在内存中，内存缓存不下才会写入磁盘。

- 通常不建议使用DISK_ONLY和后缀为_2的级别：因为完全基于磁盘文件进行数据的读写，会导致性能急剧降低，有时还不如重新计算一次所有RDD。后缀为_2的级别，必须将所有数据都复制一份副本，并发送到其他节点上，数据复制以及网络传输会导致较大的性能开销，除非是要求作业的高可用性，否则不建议使用。

## 使用Cache注意下面三点
- cache之后一定不能立即有其它算子，不能直接去接算子。因为在实际工作的时候，cache后有算子的话，它每次都会重新触发这个计算过程。

- cache不是一个action，运行它的时候没有执行一个作业。

- cache缓存如何让它释放缓存：unpersist，它是立即执行的。persist是lazy级别的（没有计算）
  
**unpersist是eager级别的。意味着unpersist如果定义在action算子之前，则cache失效**
