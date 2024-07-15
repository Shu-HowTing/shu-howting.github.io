+++
title = "Spark各种概念理解"
date = "2019-09-01"
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

## Spark中的各种概念的理解

Application：通俗讲，用户每次提交的所有的代码为一个application。

Job：一个application可以分为多个job。如何划分job？通俗讲，触发一个final RDD的实际计算（**action**）为一个job

Stage：一个job可以分为多个stage。根据一个job中的RDD的宽依赖和窄依赖关系进行划分

Task：task是最小的基本的计算单位。一般是一个RDD的一个分区（partition）为一个task，大约是128M


并行度：是指指令并行执行的最大条数。在指令流水中，同时执行多条指令称为指令并行。

理论上：每一个stage下有多少的分区，就有多少的task，task的数量就是我们任务的最大的并行度。

（一般情况下，我们一个task运行的时候，使用一个cores）

实际上：最大的并行度，取决于我们的application任务运行时使用的executor拥有的cores的数量。

[spark 基本概念解析](http://litaotao.github.io/spark-questions-concepts?s=inner)

### Spark运行流程
1. Application 首先被 Driver 构建 DAG 图并分解成 Stage。
2. 然后 Driver 向 Cluster Manager 申请资源。
3. Cluster Manager 向某些 Work Node 发送征召信号。
4. 被征召的 Work Node 启动 Executor 进程响应征召，并向 Driver 申请任务。
5. Driver 分配 Task 给 Work Node。
6. Executor 以 Stage 为单位执行 Task，期间 Driver 进行监控。
7. Driver 收到 Executor 任务完成的信号后向 Cluster Manager 发送注销信号。
8. Cluster Manager 向 Work Node 发送释放资源信号。
9. Work Node 对应 Executor 停止运行。

![image](https://s2.ax1x.com/2019/08/29/mLRiZt.jpg)