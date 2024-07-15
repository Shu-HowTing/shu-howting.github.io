+++
title = "Flink流处理入门"
date = "2020-08-25"
description = "Flink"
tags = [
  "flink"
]
categories = [
  "flink"
]

draft = false
math = true
mathjax = true
+++

## Flink 流处理API 

![image-20210630103508004](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/flink-flow.png)

### 1. Environment

#### 1.1 getExecutionEnvironment

> 创建一个执行环境，表示当前执行程序的上下文。如果程序是独立调用的，则此方法返回本地执行环境；如果从命令行客户端调用程序以提交到集群，则此方法返回此集群的执行环境，也就是说，getExecutionEnvironment会根据查询运行的方式决定返回什么样的运行环境，是最常用的一种创建执行环境的方式。

```java
ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
```

#### 1.2 createLocalEnvironment

```java
LocalStreamEnvironment env = StreamExecutionEnvironment.createLocalEnvironment(1);
```

#### 1.3 createRemoteEnvironment

```java
StreamExecutionEnvironment env = StreamExecutionEnvironment.createRemoteEnvironment("jobmanage-hostname", 6123,"YOURPATH//WordCount.jar");
```

### 2. Source

#### 2.1 集合

```java
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
env.setParallelism(1);

// 从集合里读取数据
DataStreamSource<SensorReading> dataStream = env.fromCollection(Arrays.asList(
    new SensorReading("sensor_1", 1547718199L, 35.8),
    new SensorReading("sensor_6", 1547718201L, 15.4),
    new SensorReading("sensor_7", 1547718202L, 6.7),
    new SensorReading("sensor_10", 1547718205L, 38.1)));

DataStreamSource<Integer> integerDataStreamSource = env.fromElements(1, 2, 4, 8);

// 打印输出
dataStream.print("data");
integerDataStreamSource.print("int");

env.execute();
```

#### 2.2 文件

```java
String path = "src/main/resources/sensor.txt";

DataStreamSource<String> dataStream = env.readTextFile(path);
```

#### 2.3 kafka消息队列

```java
Properties properties = new Properties();
properties.setProperty("bootstrap.servers", "localhost:9092");
properties.setProperty("group.id", "consumer-group");
properties.setProperty("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
properties.setProperty("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
properties.setProperty("auto.offset.reset", "latest");


//从kafka读取文件
DataStreamSource<String> sensor = env.addSource(new FlinkKafkaConsumer011<String>("sensor", new SimpleStringSchema(), 			
                                                                                  properties));
```

#### 2.4 自定义Source

> 除了以上的source数据来源，我们还可以自定义source。需要做的，只是传入一个SourceFunction就可以。具体调用如下：

```java
DataStreamSource<SensorReading> dataStream = env.addSource(new MySourceFucntion());
```

> 我们希望可以随机生成传感器数据，MySensorSource具体的代码实现如下：

```java
public static class MySourceFucntion implements SourceFunction<SensorReading>{
    private boolean running = true;

    @Override
    public void run(SourceContext<SensorReading> ctx) throws Exception {
        HashMap<String, Double> sensorTempMap = new HashMap<>(10);
        // 设置10个传感器的初始温度
        Random random = new Random();
        for (int i = 0; i < 10; i++) {
            sensorTempMap.put("sensor_" + (i+1), 60 + random.nextGaussian() * 20);
        }

        while(running){
            for (String sensorId : sensorTempMap.keySet() ) {
                // 在当前温度基础上随机波动
                double newTemp = sensorTempMap.get(sensorId) + random.nextGaussian();
                sensorTempMap.put(sensorId, newTemp);
                ctx.collect(new SensorReading(sensorId, System.currentTimeMillis(), newTemp));
            }
            // 控制输出频率
            Thread.sleep(1000L);
        }
    }

    @Override
    public void cancel() {
        running = false;
    }
}
```

### 3. Transform



#### 3.1 map

![image-20210630084715606](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/Flink-map.png)

```java
DataStreamSource<String> inputStream = env.readTextFile(path);
...
DataStream<Integer> mapStram = inputStream.map(new MapFunction<String, Integer>() {
        public Integer map(String value) throws Exception {
            return value.length();
        }
    });
```

#### 3.2 flatMap

```java
// 2. flatmap，按逗号分字段
DataStream<String> flatMapStream = inputStream.flatMap(new FlatMapFunction<String, String>() {
    @Override
    public void flatMap(String value, Collector<String> out) throws Exception {
        String[] fields = value.split(",");
        for( String field: fields )
            out.collect(field);
    }
});
```

#### 3.3 filter

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/flink_filter.png)

```java
DataStream<String> filterStream = inputStream.filter(line -> line.startsWith("sensor_1"));
```

#### 3.4 keyBy

![image-20210630084918876](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/flink-keyBy.png)

>  **DataStream**→**KeyedStream**：逻辑地将一个流拆分成不相交的分区，每个分区包含具有相同key的元素，在内部以hash的形式实现的.

#### 3.5 滚动聚合算子(Rolling Aggregation)

​	这些算子可以针对$KeyedStream$的每一个支流做聚合。

- sum()
- min()
- max()
- minBy()
- Maxby()

#### 3.6 Reduce

> **KeyedStream** →**DataStream**：一个分组数据流的聚合操作，合并当前的元素和上次聚合的结果，产生一个新的值，返回的流中包含每一次聚合的结果，而不是只返回最后一次聚合的最终结果。

```java
DataStreamSource<String> inputStream = env.readTextFile(path);

        DataStream<SensorReading> mapStream = inputStream.map(line -> {
            String[] fields = line.split(",");
            return new SensorReading(fields[0], Long.parseLong(fields[1]), Double.parseDouble(fields[2]));
        });
        KeyedStream<SensorReading, Tuple> keyedStream = mapStream.keyBy("id");

        SingleOutputStreamOperator<SensorReading> reduceStream = keyedStream.reduce(new ReduceFunction<SensorReading>() {
            @Override
            public SensorReading reduce(SensorReading value1, SensorReading value2) throws Exception {
                return new SensorReading(value1.getId(), value2.getTimestamp(),
                        Math.max(value1.getTemperature(), value2.getTemperature()));
            }
        });

        //SensorReading{id='sensor_1', timestamp=1547718207, temperature=36.3}
        //SensorReading{id='sensor_1', timestamp=1547718209, temperature=36.3}
        reduceStream.print();
        env.execute();
```

#### 3.7 Split/Select

![image-20210630100205168](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/flink-split.png)

**DataStream** → **SplitStream**：根据某些特征把一个$DataStream$拆分成两个或者多个$DataStream$.

![image-20210630100619233](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/flink-select.png)

**SplitStream**→**DataStream**：从一个SplitStream中获取一个或者多个DataStream.

```java
//传感器数据按照温度高低（以 30 度为界），拆分 成两个流 。
DataStreamSource<String> inputStream = env.readTextFile(path);

DataStream<SensorReading> dataStream = inputStream.map(line -> {
    String[] fields = line.split(",");
    return new SensorReading(fields[0], Long.parseLong(fields[1]), Double.parseDouble(fields[2]));
});
// 分流，按照温度值30度为界分为两条流
SplitStream<SensorReading> splitStream = dataStream.split(new OutputSelector<SensorReading>() {
    @Override
    public Iterable<String> select(SensorReading value) {
        return (value.getTemperature() > 30) ? Collections.singletonList("high") : Collections.singletonList("low");
    }
});

DataStream<SensorReading> highStream = splitStream.select("high");
DataStream<SensorReading> lowStream = splitStream.select("low");
```

#### 3.8 Connect和CoMap

![image-20210630102511107](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/flink-connect.png)

> **DataStream,DataStream** → **ConnectedStreams**：连接两个保持他们类型的数据流，两个数据流被Connect之后，只是被放在了一个同一个流中，内部依然保持各自的数据和形式不发生任何变化，两个流相互独立。

![image-20210630102609192](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/flink-CoMap.png)

> **ConnectedStreams → DataStream**：作用于$ConnectedStreams$上，功能与map和flatMap一样，对ConnectedStreams中的每一个Stream分别进行map和flatMap处理

```java
// 2. 合流 connect，将高温流转换成二元组类型，与低温流连接合并之后，输出状态信息
DataStream<Tuple2<String, Double>> warningStream = highStream.map(new MapFunction<SensorReading, Tuple2<String, Double>>() {
            @Override
            public Tuple2<String, Double> map(SensorReading value) throws Exception {
                return new Tuple2<>(value.getId(), value.getTemperature());
            }
        });
ConnectedStreams<Tuple2<String, Double>, SensorReading> connectedStreams = warningStream.connect(lowStream);

SingleOutputStreamOperator<Object> resultStream = connectedStreams.map(new CoMapFunction<Tuple2<String, Double>, SensorReading, 		Object>() {
                @Override
                public Object map1(Tuple2<String, Double> value) throws Exception {
                    return new Tuple3<>(value.f0, value.f1, "high temperature warning");
                }

                @Override
                public Object map2(SensorReading value) throws Exception {
                    return new Tuple2<>(value.getId(), "normal");
                }
            });
```

#### 3.9 Union

![image-20210630103112209](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/flink-union.png)

> **DataStream** → **DataStream**：对两个或者两个以上的$DataStream$进行union操作，产生一个包含所有$DataStream$元素的新$DataStream$。
>
> - Union之前两个流的类型必须是一样，Connect可以不一样，在之后的coMap中再去调整成为一样的。
>
> - Connect只能操作两个流，Union可以操作多个

```java
highStream.union(lowStream, allStream);
```

### 总览

![image-20210630103112209](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/flink-transform.png)
