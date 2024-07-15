+++
title = "Kafka入门"
date = "2020-09-02"
description = "Kafka概述"
tags = [
  "kafka"
]
categories = [
  "kafka"
]

draft = false
math = true
mathjax = true
+++

## Kafka基本概念

>  Kafka 是一个分布式的基于发布/订阅模式的消息队列（Message Queue），主要应用于大数据实时处理领域。

## 1. 消息队列(MQ)

### 1.1 优点

- 解耦
- 削封
- 缓冲
- 异步通信

### 1.2 两种模式

- 点对点(一对一，消费者主动拉取数据，消息收到后消息清除)

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/kafka-1.png)
  ​       消息生产者生产消息发送到Queue中，然后消息消费者主动从Queue中取出并且消费消息。消息被消费以后，queue中不再有存储，所以消息消费者不可能消费到已经被消费的消息。Queue支持存在多个消费者，但是对一个消息而言，只会有一个消费者可以消费。

- 发布/订阅模式（一对多，消费者消费数据之后不会清除消息）

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/image-kafka-2.png)

  消息生产者（发布）将消息发布到topic中，同时有多个消息消费者（订阅）消费该消息。和点对点方式不同，发布到topic的消息会被所有订阅者消费。

  发布/订阅模式既可以通过队列向消费者推送(类似于微信公众号); 也可以消费者去主动拉取(kafka),这样可以由消费者自己决定消费的速度，但同时，要维护一个长轮询，去监听队列中是否有新消息到达，造成一定的资源浪费。

## 2. Kafka的基础架构

![]("https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/kafka-3.jpg)

- **Producer** ：消息生产者，就是向kafka broker 发消息的客户端；
- **Consumer** ：消息消费者，向kafka broker 取消息的客户端；
- **Consumer Group** （**CG**）：消费者组，由多个consumer 组成。消费者组内每个消费者负责消费不同分区的数据，一个分区只能由一个组内消费者消费；消费者组之间互不影响。所有的消费者都属于某个消费者组，即消费者组是逻辑上的一个订阅者。
- **Broker** ：一台kafka 服务器就是一个broker。一个集群由多个broker 组成。一个broker 可以容纳多个topic。
- **Topic** ：可以理解为一个队列，生产者和消费者面向的都是一个**topic**；
- **Partition**：为了实现扩展性，一个非常大的topic 可以分布到多个broker（即服务器）上，一个**topic** 可以分为多个**partition**，每个partition 是一个有序的队列；
- **Replica**：副本，为保证集群中的某个节点发生故障时，该节点上的partition 数据不丢失.且kafka仍然能够继续工作，kafka提供了副本机制，一个topic的每个分区都有若干个副本，一个**leader**和若干个**follower**
- **leader**：每个分区多个副本的“主”，生产者发送数据的对象，以及消费者消费数据的对象都是leader.
- **follower**：每个分区多个副本中的“从”，实时从leader中同步数据，保持和leader数据的同步。leader发生故障时，某个follower会成为新的follower。

## 3. Kafka的常用指令

```bash
#创建topic(副本数不能大于broker的个数)  --partitions:分区数  --replication-factor：副本数(不能大于brokers的个数)
bin/kafka-topics.sh --create --topic white --bootstrap-server hadoop101:9092 --partitions 3 --replication-factor 3
# 查看topic
bin/kafka-topics.sh  --list  --bootstrap-server hadoop101:9092  #列出可用的topics
bin/kafka-topics.sh  --describe  --bootstrap-server hadoop101:9092 --topic white #查看具体topic的信息
# 删除topic
bin/kafka-topics.sh  --delete --bootstrap-server hadoop101:9092 --topic white #标记删除，过一段时间kafka会自己删除
# 修改topic
bin/kafka-topics.sh  --alter  --bootstrap-server hadoop101:9092 --topic white --partitions 6 #修改topic的分区数

# 生产者
bin/kafka-console-producer.sh --topic white --broker-list hadoop101:9092 
# 消费者
bin/kafka-console-consumer.sh --topic white --bootstrap-server hadoop101:9092
# 从头开始消费
bin/kafka-console-consumer.sh --topic white --bootstrap-server hadoop101:9092 --from-beginning

# 查看log数据
# ***.index文件(offset + 物理位置)
bin/kafka-dump-log.sh --files /opt/module/kafka-2.4.1/datas/white-0/00000000000000000000.index --print-data-log
# ***.log文件(真实的数据)
bin/kafka-dump-log.sh --files /opt/module/kafka-2.4.1/datas/white-0/00000000000000000000.log --print-data-log
```

