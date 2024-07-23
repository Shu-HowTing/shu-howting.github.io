+++
title = "FM算法原理"
date = "2021-06-14"
description = "FM的算法原理"
tags = [
  "FM"
]
categories = [
  "machine learning"
]

draft = false
math = true
mathjax = true
+++

## FM的提出
LR为普通的线性模型，优点是复杂度低、方便求解，但缺点也很明显，没有考虑特征之间的交叉，表达能力有限。

$$
y=\omega_0+\sum_{i=1}^n \omega_i x_i
$$

FM在线性模型的基础上添加了一个多项式，用于描述特征之间的二阶交叉:

$$
y=\omega_0+\sum_{i=1}^n \omega_i x_i+\sum_{i=1}^{n-1} \sum_{j=i+1}^n \omega_{i j} x_i x_j
$$

其中，$n$代表样本的特征数量，$x_i$是第$i$个特征的值， $w_0, w_i, w_{ij}$是模型参数。

### 问题
参数 $w_{i j}$ 学习困难, 因为对 $w_{i j}$ 进行更新时, 求得的梯度对应为 $x_i x_j$, 当且仅当 $x_i$ 与 $x_j$ **都非0时参数才会得到更新**。
但是经过 one-hot 处理的数据非常稀疏，能够保证两者都非 0 的组合较少，导致大部分参数$w_{i j}$难以得到充分训练。

### 解决方案
对每个特征分量 $x_i$ 引入 $k$ 维 $($k \<< n$)$ 辅助向量 $v_i=\left(v_{i 1}, v_{i 2}, \ldots, v_{i k}\right)$, 每个特征对应一个$k$维的emb,总共 $n$ 个emb, 然后利用向量内积的结果 $<v_i, v_j>$ 来表示原来的组合参数 $w_{i j}$.

于是，原式变成了如下形式：(尖括号表示内积)
$$
\hat{y}(\mathbf{x}):=w_0+\sum_{i=1}^n w_i x_i+\sum_{i=1}^n \sum_{j=i+1}^n\left\langle\mathbf{v}_i, \mathbf{v}_j\right\rangle x_i x_j
$$

这样要学习的参数从$n(n-1)/2$个$w_{ij}$系数变成了元素个数为$n\times k$的$V$矩阵，因为$k\<<n$,所以降低了训练复杂度。

$$
\mathbf{V}=\left(\begin{array}{cccc}
v\_{11} & v\_{12} & \cdots & v\_{1 k} \\\\
v\_{21} & v\_{22} & \cdots & v\_{2 k} \\\\
\vdots & \vdots & & \vdots \\\\
v_{n 1} & v_{n 2} & \cdots & v\_{n k}
\end{array}\right)_{n \times k}
=\left(\begin{array}{c}
\mathbf{v}\_1 \\\\
\mathbf{v}\_2 \\\\
\vdots \\\\
\mathbf{v}\_{n}
\end{array}\right)
$$

> 此外，引入辅助向量削弱了参数间的独立性，因为对于$x_i$的隐向量$v_i$ ,任何包含$x_i$的特征组合， 只要$x_i$本身不为0，都可对$v_i$进行更新，同理每个隐向量都能得到充分的学习，这样就解决了数据稀疏带来的难以训练问题。


### 运算简化

\begin{equation}
    \begin{aligned}
      \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}<v_{i},v_{j}>x_{i}x_{j}& =\frac12\sum_{i=1}^{n}\sum_{j=1}^{n}<v_{i},v_{j}>x_{i}x_{j}-\frac12\sum_{i=1}^{n}<v_{i},v_{i}>x_{i}x_{i} \\\\
      &=\frac{1}{2}\left(\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{f=1}^{k}v_{i,f}v_{j,f}x_{i}x_{j}-\sum_{i=1}^{n}\sum_{f=1}^{k}v_{i,f}v_{i,f}x_{i}x_{i}\right) \\\\
      &=\frac12\sum_{f=1}^k\left[\left(\sum_{i=1}^nv_{i,f}x_i\right)\cdot\left(\sum_{j=1}^nv_{j,f}x_j\right)-\sum_{i=1}^nv_{i,f}^2x_i^2\right] \\\\
      &=\frac12\sum_{f=1}^k\left[\left(\sum_{i=1}^nv_{i,f}x_i\right)^2-\sum_{i=1}^nv_{i,f}^2x_i^2\right]
    \end{aligned}
\end{equation}

> 参考: $ab+ac+bc=\frac{1}{2}\left[(a+b+c)^2-(a^2+b^2+c^2)\right]$

对需要训练的参数$\theta$求梯度得：

$$
\begin{equation}
    \frac{\partial\hat{y}(x)}{\partial\theta}=\begin{cases} 1,&if~\theta~is~\omega_0 \\\\ 
	x_i,&if~\theta~is~\omega_i\\\\
	x_i\sum_{j=1}^nv_{j,f}x_j-v_{i,f}x_i^2 &if~\theta~is~v_{i,f}\end{cases}
\end{equation}
$$
重点关注$v_{if}$的梯度，$v_{if}$表示$x_i$的隐向量，因为梯度项$\sum_{j=1}^{n} v_{j,f}x_j$中不包含$i$ ,只与$f$有关，因此只要一次性求出所有的$f$的$\sum_{j=1}^nv_{j,f}x_j$的值 (复杂度$O(nk))$,在求每个参数的梯度时都可复用该值。

当已知$\sum_{j=1}^nv_{j,f}x_j$ 时计算每个参数梯度的复杂度都是 $O(1)$ ,因此训练 FM 模型的复杂度也是$O(nk)$。


参考代码:
```python
import tensorflow as tf
import tensorflow.keras.backend as K

class FM_layer(tf.keras.layers.Layer):
    def __init__(self, k, w_reg, v_reg):
        super(FM_layer, self).__init__()
        self.k = k   # 隐向量vi的维度
        self.w_reg = w_reg  # 权重w的正则项系数
        self.v_reg = v_reg  # 权重v的正则项系数

    def build(self, input_shape): 
		# shape:(1,)
        self.w0 = self.add_weight(name='w0', shape=(1,), 
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
		# shape:(n, 1)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1), 
                                 initializer=tf.random_normal_initializer(), 
                                 trainable=True, 
                                 regularizer=tf.keras.regularizers.l2(self.w_reg)) 
        # shape:(n, k)
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        # inputs维度判断，不符合则抛出异常
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        # 线性部分，相当于逻辑回归  (B, 1)
        linear_part = tf.matmul(inputs, self.w) + self.w0   
        # 交叉部分——第一项  (B, k)
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  
        # 交叉部分——第二项 (B, k)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2)) 
        # 交叉结果 (B, 1)
        inter_part = 0.5*tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True) 

        output = linear_part + inter_part
        return tf.nn.sigmoid(output) 
```