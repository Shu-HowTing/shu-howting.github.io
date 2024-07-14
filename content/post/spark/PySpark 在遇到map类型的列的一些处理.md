+++
title = "Spark map字段处理"
date = "2021-09-01"
description = "MapReduce"
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

## $PySpark$ 在遇到$map$类型的列的一些处理

> 在$spark$中，有时会遇到$column$的类型是$array$和$map$类型的，这时候需要将它们转换为多行数据

### $Explode\ array\ and\ map\ columns\ to\ rows$

```python
import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('pyspark-by-examples').getOrCreate()

arrayData = [
        ('James',['Java','Scala'],{'hair':'black','eye':'brown'}),
        ('Michael',['Spark','Java',None],{'hair':'brown','eye':None}),
        ('Robert',['CSharp',''],{'hair':'red','eye':''}),
        ('Washington',None,None),
        ('Jefferson',['1','2'],{}) ]

df = spark.createDataFrame(data=arrayData, schema = ['name','knownLanguages','properties'])
df.printSchema()
df.show()
```

```markdown
root
 |-- name: string (nullable = true)
 |-- knownLanguages: array (nullable = true)
 |    |-- element: string (containsNull = true)
 |-- properties: map (nullable = true)
 |    |-- key: string
 |    |-- value: string (valueContainsNull = true)

+----------+--------------+--------------------+
|      name|knownLanguages|          properties|
+----------+--------------+--------------------+
|     James| [Java, Scala]|[eye -> brown, ha...|
|   Michael|[Spark, Java,]|[eye ->, hair -> ...|
|    Robert|    [CSharp, ]|[eye -> , hair ->...|
|Washington|          null|                null|
| Jefferson|        [1, 2]|                  []|
+----------+--------------+--------------------+
```

#### $explode – array\ column\ example$

> $PySpark\ function$ `explode(e: Column)` is used to explode or create array or map columns to rows. When an array is passed to this function, it creates a new default column “col1” and it contains all array elements. When a map is passed, it creates two new columns one for key and one for value and each element in map split into the rows.


> $spark$提供$explode$函数`explode(e: Column)`， 当传入的column是array类型时，它会新建一个列，默认列名为`col`；当传入的column是map类型时，则会新建两个列，一个列为key，另一个为value

```python
from pyspark.sql.functions import explode
df3 = df.select(df.name, explode(df.knownLanguages))
df3.printSchema()
df3.show()
```

##### $output$

```

root
 |-- name: string (nullable = true)
 |-- col: string (nullable = true)

+---------+------+
|     name|   col|
+---------+------+
|    James|  Java|
|    James| Scala|
|  Michael| Spark|
|  Michael|  Java|
|  Michael|  null|
|   Robert|CSharp|
|   Robert|      |
|Jefferson|     1|
|Jefferson|     2|
+---------+------+
```

> **注意：**
>
> `Washington`对应的$knownLanguages$字段是null，explode会忽略这种值，可以看到，结果集里并没有`Washington`的记录，如果需要保留，使用`explode_outer`函数

#### $explode – map\ column\ example$

```python
from pyspark.sql.functions import explode
df3 = df.select(df.name,explode(df.properties))
df3.printSchema()
df3.show()
```

##### $output$

```
root
 |-- name: string (nullable = true)
 |-- key: string (nullable = false)
 |-- value: string (nullable = true)

+-------+----+-----+
|   name| key|value|
+-------+----+-----+
|  James| eye|brown|
|  James|hair|black|
|Michael| eye| null|
|Michael|hair|brown|
| Robert| eye|     |
| Robert|hair|  red|
+-------+----+-----+
```

### $How\ to\ covert\ Map\ into\ multiple\ columns$

> 有时候需要把$Map$类型的$colum$n进行以$key$为列名，$value$为列值的处理。如下：

```python
from pyspark.sql import functions as F

df.select(F.col("name"),
   F.col("properties").getItem("hair").alias("hair_color"),
   F.col("properties").getItem("eye").alias("eye_color")).show()
```

##### $output$

```
+----------+----------+---------+
|      name|hair_color|eye_color|
+----------+----------+---------+
|     James|     black|    brown|
|   Michael|     brown|     null|
|    Robert|       red|         |
|Washington|      null|     null|
| Jefferson|      null|     null|
+----------+----------+---------+
```







