+++
# author = "丁树浩"
title = "当推荐遇到大模型"
date = "2024-07-01"
slug = "seq_llm_model"
description = "大模型在推荐领域的应用"
tags = [
  "推荐",
  "大模型"
]
categories = [
  "RecSys"
]
# image = "3.jpg"
draft = false
math = true
+++


> 自从大语言模型爆火之后，大家对大语言模型（LLM）如何成功应用在推荐系统进行了不少尝试。本文是对目前一些业界工作的调研和总结。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/5bb122b8-edef-4175-8167-8aa1524cd7df.png)

# 大模型应用范式

经典的推荐架构基本遵循以下范式：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/pgEG19GKDHxeolWl/27da04c5ef2c47c7a7642a1a201f15f53264.png)

目前, LLM 在推荐系统中的主流应用可以分为两种范式:

* 一个是作为经典推荐系统的辅助部分，即 **LLM+RS**。
* 一个是 LLM 单独作为一个完整的推荐系统，即 **LLM AS RS**。
    

本文接下来将分别介绍这两种应用方式。

## 2.1 LLM+RS

> **传统推荐系统经过多年发展，从召回、排序、重排到最终展示的架构已经比较成熟。LLM+RS 是将 LLM 作为推荐链路的一部分，来影响召回、排序等环节。LLM 影响推荐系统的方式多种多样。主要有以下几种:**

* **利用大模型结构强大的学习能力，直接替换现有推荐模型的结构，如利用transformer进行序列建模等**  
* **利用大模型的表征能力，生成推荐物料的文本或图像的表征向量/token，作为现有推荐模型的输入**
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/f1b83414-29bc-4699-97ac-82d7231c6c56.png)

下面我们从上述方向出发，介绍每一个方向的代表工作。

### 2.1.1 利用大模型进行模型结构升级

这部分最典型的工作集中在推荐中的序列特征方面。因为序列特征天然和NLP的token输入天然具有相似性。经典的有BST， SASRec。

#### BST

BST采用的是Transformer中的Encoder部分的结构进行序列特征的处理。结构如下图：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/f16de178-f64e-4421-a755-76e559f413b4.png)

#### SASRec 

SASRec借鉴了Transformer中Decoder部分的结构，输入的是用户行为序列，不断预测下一个用户交互的item(类似GPT)：
![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/fffe4a40-d715-4339-9ec6-a98018510348.png)

#### BERT4Rec
![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/ae060b3f-afad-4d7e-b2f7-71cd27a523e0.png)

### 2.1.2 利用大模型进行向量/token表征

#### 独立于现有推荐模型

##### NoteLLM
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/f28f25ad-7ad9-43ed-ab0d-dc84e00a7c89.png)

 输入NoteLLM的Prompt的格式模板如下:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/5ed99d91-04f8-4684-8534-869816f2fa44.png)

其中, \[BOS\], \[EMB\]和\[EOS\]为特殊token, 而<Instruction>, <Input Note>, <Output Guidance>和<Output>为占位符, 对于不同的任务会使用不同特定的内容来替换。

- **类别生成任务**的笔记压缩提示模板如下:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/0d3d18aa-4b62-4bf0-a197-5fddb0792a02.png)

-  **主题标签生成任务**的笔记压缩提示模板如下:
    
![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/c8d02652-0d2c-4167-866b-a4742fc544ce.png)

学习目标分为两部分:
- 无监督的对比学习
- 有监督的category / topic预测
    

一个完整的prompt case：

```
Extract the note information in json format, compress it into one word for recommendation, and generate the category of the note.

{'title': 'How to Train Your Dragon', 'topic': 'Movies', 'content': 'A young Viking forms a bond with a dragon.'}

The compression word is:"\[EMB\]".

The category is Fantasy.
```

所以， loss为：

$$L=\frac{L\_{cl}+\alpha L\_{gen}}{1+\alpha}$$

其中：

$$L\_{cl}=-\frac{1}{2B}\sum\_{i=1}^{2B}log\frac{e^{sim(\boldsymbol{n}\_{i},\boldsymbol{n}\_{i}^{+})\cdot e^{\tau}}}{\sum\_{j\in\[2B\]\setminus\{i\}}e^{sim(\boldsymbol{n}\_{i},\boldsymbol{n}\_{j})\cdot e^{\tau}}}$$

$$L\_{gen}=-\frac{1}{T}\sum\_{i=1}^{T}log(p(o\_{i}|o\_{<i},i))$$

#### 联合现有推荐模型训练：

##### CTRL
![111image-topaz-enhance-1.7x.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/83027468-a2a2-4deb-8f4a-60a3b42c36a9.png)

**两阶段训练：**

- step1: 语言模型(LLM)和推荐模型进行无监督的对比学习

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/53e18d76-9b26-4bf1-a107-9f8823efbcee.png)

- step2: 推荐模型单独进行有监督的微调训练

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/4589882f-b81d-4802-9459-ed6fc0056ca3.png)

## 2.2 LLM AS RS

> **LLM 作为推荐系统，与 LLM+RS 最大的区别在于，它不再作为一个部分来影响推荐系统，而是以端到端的方式使用一个大模型作为整个系统，LLM 将直接面对用户和商品。**

根据替换程度不同，我们也可以分为两种情况:

- **局部替换: 指大模型将代替推荐流程的某一环。比如精排环节, 采用prompt方式，直接让大模型从召回集合中输出排序推荐结果**
- **整体替换: 推翻现有的经典推荐架构, 直接使用大模型end2end进行推荐。**
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/4505479c-28bf-46ee-b597-ccaf2b007137.png)

### 2.2.1 局部替换：

比如用大模型代替原来的排序模块对召回的结果进行打分排序。

- \[Google\] Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/dbf55ed1-a73d-4978-b0d7-702aff5eedfe.png)

- \[Amazon\] PALR: Personalization Aware LLMs for Recommendation
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/0c859c04-e9ef-43a0-aef8-ce600d60b38d.png)

**prompt:**

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/3edda769-4972-4567-b372-4a1f6c48ad30.png)

### 2.2.2 整体替换：

彻底颠覆现有的经典架构，用大模型进行end2end的训练和推荐预测。

- \[Meta\] Actions Speak Louder than Words 

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/da5f7f8e-7212-4c74-ba0b-b9f86da86fe7.png)

**补充：**

目前，根据训练和推理阶段的不同做法，可以从如下四个角度区分现阶段的研究方向：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/KM7qe92bYQ5eqpj8/img/ce22c427-fdd4-46e6-b48a-dffff63fd854.png)

*   在**训练**阶段，大语言模型是否需要**微调**。这里微调的定义包含了全量微调和参数高效微调。
*   在**推理**阶段，是否需要**引入传统推荐模型**(CRM)。其中，如果CRM知识作为一个预先过滤candidate的作用，则不被考虑在内。
    

## Referrence

1. [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1905.06874)
2. [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/pdf/1904.06690)
3. [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/pdf/1904.06690) 
4. [A Survey on Large Language Models for Recommendation](https://arxiv.org/pdf/2305.19860) 
5. [How Can Recommender Systems Benefit from Large Language Models: A Survey](https://arxiv.org/pdf/2306.05817)
6. [CTRL: Connect Collaborative and Language Model for CTR Prediction](https://arxiv.org/pdf/2306.02841)
7. [NoteLLM: A Retrievable Large Language Model for Note Recommendation](https://arxiv.org/pdf/2403.01744)
8. [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/pdf/2402.17152)
9. [WWW'24 | 小红书NoteLLM: 大语言模型用于笔记推荐](https://zhuanlan.zhihu.com/p/698568773)
10. [PALR: Personalization Aware LLMs for Recommendation](https://arxiv.org/pdf/2305.07622)