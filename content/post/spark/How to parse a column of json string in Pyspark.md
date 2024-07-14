+++
title = "Spark解析DataFrame中的json字段"
date = "2021-09-01"
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

## How to parse a column of json string in Pyspark

> 在用$spark.sql(\ )$从Table读入数据时，`DataFrame`的列有时是这样一种类型：`json`形式的`string`。此时，我们通常需要去解析这个json string，从而提取我们想要的数据。

### 数据准备

```python
# Sample Data Frame
jstr1 = u'{"header":{"id":12345,"foo":"bar"},"body":{"id":111000,"name":"foobar","sub_json":{"id":54321,"sub_sub_json":{"col1":20,"col2":"somethong"}}}}'
jstr2 = u'{"header":{"id":12346,"foo":"baz"},"body":{"id":111002,"name":"barfoo","sub_json":{"id":23456,"sub_sub_json":{"col1":30,"col2":"something else"}}}}'
jstr3 = u'{"header":{"id":43256,"foo":"foobaz"},"body":{"id":20192,"name":"bazbar","sub_json":{"id":39283,"sub_sub_json":{"col1":50,"col2":"another thing"}}}}'
df = spark.createDataFrame([Row(json=jstr1),Row(json=jstr2),Row(json=jstr3)])
```

> 如上所示，我们模拟一个DataFrame，其中只有一列，列名为`json`，类型为`string`。可以看到，`json`中的值为json格式。我们如何从中取出我们关心的值，形成一个单独的列呢？例如：$df['header']['id']$.

### from_json函数

```python
from pyspark import Row
from pyspark.sql.functions import from_json, col

json_schema = spark.read.json(df.select('json').rdd.map(lambda row: row.json)).schema
df_json = df.withColumn('json', from_json(col('json'), json_schema))
print(json_schema)
```

**$Result:$**

```markdown
root
 |-- body: struct (nullable = true)
 |    |-- id: long (nullable = true)
 |    |-- name: string (nullable = true)
 |    |-- sub_json: struct (nullable = true)
 |    |    |-- id: long (nullable = true)
 |    |    |-- sub_sub_json: struct (nullable = true)
 |    |    |    |-- col1: long (nullable = true)
 |    |    |    |-- col2: string (nullable = true)
 |-- header: struct (nullable = true)
 |    |-- foo: string (nullable = true)
 |    |-- id: long (nullable = true)
```

```python
df_json.select(col('json.header.id').alias('id')).show()
```

**$Result:$**

```markdown
+-----+
|   id|
+-----+
|12345|
|12346|
|43256|
+-----+
```

```python
df_json.select(col('json.header.id').alias('id'), col('json.body.name').alias('name')).show()
```

**$Result:$**

```markdown
+-----+-------+
|   id|   name|
+-----+-------+
|12345| foobar|
|12346| barfoo|
|43256| bazbar|
+-----+-------+
```

[参考链接](https://stackoverflow.com/questions/41107835/pyspark-parse-a-column-of-json-strings)

