+++
title = "Transformer模型理解"
date = "2021-11-15"
description = "Transformer"
tags = [
  "transformer"
]
categories = [
  "deep learning"
]

draft = false
math = true
mathjax = true
+++

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/202407171131481.png)

> Transformer模型在2017年被google提出，直接基于Self-Attention结构，并且迅速取代了之前NLP任务中常用的RNN神经网络结构，成为主流。本文将探讨关于transformer模型的实现细节

## Transformer

### Encoder

#### Self-attention
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/self-att1.png)
![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/self-att2.png)

Transformer 中token的输入表示$a$由$Word\ Embedding$ 和位置 $Positional\ Encoding$ 相加得到。

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/202407171418748.png)
#### Add & Norm

Add & Norm 层由 Add 和 Norm 两部分组成，其计算公式如下：
$$\textit{LayerNorm}\big(X+\text{MultiHeadAttention}(X)\big)$$


#### Feed Forward
Feed Forward 层比较简单，是一个两层的全连接层，第一层的激活函数为 Relu，第二层不使用激活函数，对应的公式如下。
$$
\max(0,XW_1+b_1)W_2+b_2
$$
需要注意的是，FF网络对于每个token是共享的，即网络参数量和序列长度无关。

### Decoder

#### Mask Attention
![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/self-att3.png)

Decoder block 的第一个 Multi-Head Attention 采用了 Masked 操作，因为在翻译的过程中是顺序翻译的，即翻译完第 i 个单词，才可以翻译第 i+1 个单词。
通过 Masked 操作可以防止第 i 个单词知道 i+1 个单词之后的信息。

#### Cross Attention
即encoder部分与decoder部分开始交互的模块, 由encoder提供 $K \\& V$, decoder提供$Q$

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/self-att4.png)


### Softmax层
Decoder block 最后的部分是利用 Softmax 预测下一个单词，在之前的网络层我们可以得到一个最终的输出 Z，因为 Mask 的存在，使得单词 0 的输出 Z0 只包含单词 0 的信息，如下：
![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/202407171420969.png)

## Transformer 总结
- Transformer 与 RNN 不同，可以比较好地并行训练。
- Transformer 本身是不能利用单词的顺序信息的，因此需要在输入中添加位置 Embedding，否则 Transformer 就是一个词袋模型了。
- Transformer 的重点是 Self-Attention 结构，其中用到的 Q, K, V矩阵通过输出进行线性变换得到。
- Transformer 中 Multi-Head Attention 中有多个 Self-Attention，可以捕获单词之间多种维度上的相关系数 attention score。

## Reference:
1. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
2. [李宏毅：transformer](https://www.youtube.com/watch?v=ugWDIIOHtPA&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&index=61)
3. [Transformer模型详解](https://zhuanlan.zhihu.com/p/338817680)