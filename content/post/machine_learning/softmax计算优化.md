+++
title = "softmax计算优化"
date = "2021-12-01"
description = "XGBoost的原理分析以及实践"
tags = [
  "softmax"
]
categories = [
  "machine learning"
]

draft = false
math = true
mathjax = true
+++

## softmax上溢和下溢问题

解决这个问题的方法就是利用softmax的冗余性。我们可以看到对于任意一个数$a$, $x-a$和$x$在$softmax$中的结果都是一样的。

$$
\frac{\exp^{(x-a)}}{\sum\_{i=1}^k \exp\_i^{(x-a)}}=\frac{\exp ^{(x)} \exp ^{(-a)}}{\exp ^{(-a)} \sum\_{i=1}^k \exp _i^{(x)}}=\frac{\exp ^{(x)}}{\sum\_{i=1}^k \exp\_i^{(x)}}
$$

对于一组输入，我们可以让a=max(x). 这样就可以保证x-a的最大值等于0，也就不会产生上溢的问题。同时，因为$x-a=0$, 所以$exp(0)=1$,分母就不可能为0。

$$
\begin{array}{l}
\log \left(\frac{\exp^{(x-a)}}{\sum\_{i=1}^k \exp\_i^{(x-a)}}\right)
&=\log \left(e^{(x-a)}\right)-\log \left(\sum\_{i=1}^k \exp\_i^{(x-a)}\right)  \\\\
&=(x-a)-\log \left(\sum\_{i=1}^k \exp\_i^{(x-a)}\right)
\end{array}
$$