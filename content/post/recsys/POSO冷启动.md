+++
# author = "丁树浩"
title = "POSO冷启动"
date = "2023-11-20"
description = ""
tags = [
  "冷启动",
  "推荐"
]
categories = [
  "RecSys"
]
draft = false
math = true
+++

## POSO
> POSO: Personalized Cold Start Modules for Large-scale Recommender Systems 是快手发表的一篇关于如何解决新老用户分群的文章。与传统的做法从样本上做文章不同，论文直接从模型结构上入手，希望模型能够区分两部分人群，达到一个新用户冷的效果。

## Motivation
文章先分析了新用户和普通用户行为的存在差异性，然后设计实验可视化了新用户特征被淹没的现象，再提出了可嵌入到主流网络结构中的方法POSO。

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/202407231439808.png)

按照常规的做法，一个模型想要hold住两种不一样的分布(新用户 && 老用户)，至少得有一个区分特征，比如$is\\_new\\_user$来进行区分。

但是，作者通过分析，发现问题没那么简单。如上图(b)所示，作者发现，把模型的$is\\_new\\_user$特征输入值进行人为改变(原来为0,该为1;原来为1,改为0。保持其他输入不变)，对模型上层的输出似乎并没有太大改变。也就是说，模型忽略了$is\\_new\\_user$特征。

是什么原因造成特征被忽略呢？本质是因为$is\\_new\\_user$是一个高度不平衡的特征，样本中只有5%以下属于新用户，即$is\\_new\\_user=1$。对于模型来说，大部分样本对应的该特征值是一样的($is\\_new\\_user=0$)，所以模型在学习上会“偷懒”，认为这个特征不重要。

那么如何解决呢，可以参考**动态权重**的做法，把这部分重要的**敏感特征**独立出来，生成一个$gate^{pc}$，直接去修正模型的网络参数, 从而更加深刻的影响模型输出。

> 作者本人不是从动态权重的角度进行解释的，而是有点类似于随即森林的想法，认为有很多基座模型，用**敏感特征**去生成基座模型的组合权重。

## 具体实现
### MLP
![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/202407191114640.png)
$$\mathrm{\hat{x}}=C\cdot g\left(\mathrm{x}^\mathrm{pc}\right)\odot\sigma\left(W\mathrm{x}\right)$$
通过$gating\ network$生成和$feature\ map$等大的masks，以$element-wise$的方式乘在每一层的$feature\ map$上.

### MMoE   
![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/202407191121236.png)

$$\hat{x}^t=\sum_ig_i^t(x)g_i(x^{pc})f_i(x) $$
先用$gate^{pc}$门控产生的权重($dim=num\\_experts$)对专家网络的输出进行修正，再用原MMoE的gate权重加权求和。

## 实践细节

### 用什么特征做gate的输入
这里重点把 uid 拿出来说，网上不少的资料里会提到使用 uid 的方案，并且论文里也有这样的测试。如果使用 uid 的话，作者建议，拷贝一份出来然后需要做$stop\ gradient$处理。

作者实验场景使用的是vv数，至少这一点上还是能够很清晰的区分新老用户，新用户肯定vv数较低，毋庸置疑。

所以最终的结论就是：选择指向性强的特征，能够明确的区分用户的特征，而且要保证分化方向的一致性。

### 分化特征如何处理

上一part说到分化特征比如vv数这样的，vv数本身是一个数值特征，该怎么喂入模型？

一般深度模型都是id类特征是一等公民，所以常见的类似vv特征都会做分桶处理，然后查embedding做不断的调整。

但是POSO不一样，$x^{pc}$的输入特征，不需要分桶进行embedding处理。而是直接输入原始的$scalar$值。


### 和MMoE的gate的区别

MMoE中也有gate，虽然两者都是学权重，但有本质差异。

- MMoE的gate是学task对expert的权重，本质是学不同task的样本的重要性，是一个预期稳定的分布，因此同一个task的不同样本，其**gate分布越稳定，某种程度表示模型收敛得越好**。
- POSO的gate是学网络参数对用户的响应，本质是学网络的个性化，是一个预期需要体现差异性的分布，因此不同样本，其**gate分布差异越大，某种程度表示模型学得越好**。
- MMoE的gate的输出是经过$softamx$的输出(保证权重之和为1.0); 但是POSO的gate的输出用的是$sigmoid$(还需要*2), 不要求权重之和为1.0

## Reference
1. [POSO方法的实际应用和分析思考](https://mp.weixin.qq.com/s?__biz=MzAxMzgzOTc2NA==&mid=2652180120&idx=1&sn=8ed33f2216192976f4accf129e819aa9&chksm=807dd263b70a5b75457da3d6f6abd0bb9fd1837c2e9d868df18a4012ba88c1f1dab51a15dba6&cur_album_id=2277635355426373635&scene=189#wechat_redirect)
2. [POSO实践的一些总结](https://www.deeplearn.me/4317.html)
3. [推荐系统难题挑战（7）：POSO，从模型角度解决用户冷启动问题](https://zhuanlan.zhihu.com/p/472726462)
4. [POSO: Personalized Cold Start Modules for Large-scale Recommender Systems](https://arxiv.org/pdf/2108.04690)