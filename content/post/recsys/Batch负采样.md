+++
# author = "丁树浩"
title = "Batch内负采样"
date = "2024-03-02"
description = ""
tags = [
  "召回",
  "推荐"
]
categories = [
  "RecSys"
]
draft = false
math = true
+++


## In-batch Negative Sampling

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/202407151807785.png)

- code:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RecommenderModel(nn.Module):
		def __init__(self, user_size, item_size, embedding_dim):
        super(RecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(user_size, embedding_dim)
        self.item_embedding = nn.Embedding(item_size, embedding_dim)

		def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        return user_embeds, item_embeds

		def in_batch_negative_sampling_loss(user_embeds, item_embeds):
		    batch_size = user_embeds.size(0)
		    
		    # 正样本得分 (batch_size,) 
		    positive_scores = torch.sum(user_embeds * item_embeds, dim=-1) 
		    
		    # 负样本得分 (batch_size, batch_size)
		    negative_scores = torch.matmul(user_embeds, item_embeds.t())  
		    
		    # 创建标签  (batch_size, batch_size)
		    labels = torch.eye(batch_size).to(user_embeds.device) 
		    
		    # 计算损失
		    loss = F.cross_entropy(negative_scores, labels.argmax(dim=-1))
		    
		    return loss

# 示例数据
batch_size = 4
embedding_dim = 8
user_size = 100
item_size = 1000

user_ids = torch.randint(0, user_size, (batch_size,))
item_ids = torch.randint(0, item_size, (batch_size,))

model = RecommenderModel(user_size, item_size, embedding_dim)
user_embeds, item_embeds = model(user_ids, item_ids)

loss = in_batch_negative_sampling_loss(user_embeds, item_embeds)
print(f'Loss: {loss.item()}')
```

### 优点

- **效性**：批量内负采样能够充分利用每个训练批次中的样本，提高训练效率，避免显式生成大量负样本的开销。
- **适用性**：这种方法特别适用于深度学习的推荐系统，在大规模数据训练时效果显著。
- **实现**：通过在每个批次中将其他正样本视为负样本，并使用合适的损失函数（如交叉熵损失），可以有效地优化模型。

### 缺点
- 对热门商品的打压过于严重
  
![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/batch_sample.png)


> batch内的item对于当前user都是正样本，这种样本天然要比随机采样的item的热度要高。也就是说，我们采样了一批热门商品当作负样本(hard negtive-sample)。这样难免对于热门商品的打压太过了，可以在计算<user, item>得分时，减去商品的热度，来补偿。

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/batch_sample2.png)



## sampled_softmax loss

$$\mathcal{L}=-log{\left[\frac{exp(s_{i,i}-log(p_j))}{\sum_{k\neq i}exp(s_{i,k}-log(p_k))+exp(s_{i,i}-log(p_j))}\right]}$$

对每个item j，假设被采样的概率为$p_j$，那么$log Q$矫正就是在本来的内积上加上 $-log{p_j}$
$$
s^c(x_i, y_j) = s(x_i, y_j) – \log {p_j}
$$

$p_j$的概率通过距离上一次看到y的间隔来估计，item越热门，训练过程中就越经常看到item，那么距离上次看到y的间隔B(y)跟概率p成反比。于是有如下算法

![](https://markdown-1258220306.cos.ap-shenzhen-fsi.myqcloud.com/img/sampel_softmax_1.png)

在实践当中，对每个y都可以用PS的1维向量来存储对应的step，那么下一次再看到y时就可以计算出对应的间隔和概率了。
```py
step = get_global_step()  # 获取当前的batch计数tensor
item_step = get_item_step_vector()   # 获取用于存储item上次见过的step的向量
item_step.set_gradient(grad=step - item_step, lr=1.0)  
delta = tf.clip_by_value(step - item_step, 1, 1000)
logq = tf.stop_gradient(tf.log(delta))
batch_logits += logq  # batch_logits 是前面计算的logits
...
```

$$
\begin{aligned}
\frac{\partial\mathcal{L}}{\partial {s_{i,j}}} \\\\
&=\frac{\exp(s_{i,j}-log(p_j))}{\sum_{k\neq i}\exp(s_{i,k}-log(p_k))+\exp(s_{i,i}-log(p_i))} \\\\
&=\frac1{p_j}P_{i,j} 
\end{aligned}
$$

> 可以看到，越热门的负样本，${1} / {p_j}$越小,gradient越小，可以起到一定的补偿机制。

```py
def sampled_softmax_loss(weights,
                         biases,
                         labels,
                         inputs,
                         num_sampled,
                         num_classes,
                         num_true=1):
    """
    weights: 待优化的矩阵，形状[num_classes, dim]。可以理解为所有item embedding矩阵，那时 num_classes = 所有item的个数
    biases: 待优化变量，[num_classes]。每个item还有自己的bias，与user无关，代表自己本身的受欢迎程度。
    labels: 正例的item ids，形状是[batch_size,num_true]的正数矩阵。每个元素代表一个用户点击过的一个item id，允许一个用户可以点击过至多num_true个item。
    inputs: 输入的[batch_size, dim]矩阵，可以认为是user embedding
    num_sampled：整个batch要采集多少负样本
    num_classes: 在u2i中，可以理解成所有item的个数
    num_true: 一条样本中有几个正例，一般就是1
    """
     # logits: [batch_size, num_true + num_sampled]的float矩阵
     # labels: 与logits相同形状，如果num_true=1的话，每行就是[1,0,0,...,0]的形式
    logits, labels = _compute_sampled_logits(
              weights=weights,
              biases=biases,
              labels=labels,
              inputs=inputs,
              num_sampled=num_sampled,
              num_classes=num_classes,
              num_true=num_true,
              sampled_values=sampled_values,
              subtract_log_q=True,
              remove_accidental_hits=remove_accidental_hits,
              partition_strategy=partition_strategy,
              name=name,
              seed=seed)
    labels = array_ops.stop_gradient(labels, name="labels_stop_gradient")
    
    # sampled_losses：形状与logits相同，也是[batch_size, num_true + num_sampled]
		# 一行样本包含num_true个正例和num_sampled个负例
		# 所以一行样本也有num_true + num_sampled个sigmoid loss

		sampled_losses = sigmoid_cross_entropy_with_logits(
		      labels=labels,
		      logits=logits,
		      name="sampled_losses")
		      
		# We sum out true and sampled losses.
		return _sum_rows(sampled_losses)

def _compute_sampled_logits(weights,
       biases,
       labels,
       inputs,
       num_sampled,
       num_classes,
       num_true=1,
       ......
       subtract_log_q=True,
       remove_accidental_hits=False,......):
    """
    输入:
        weights: 待优化的矩阵，形状[num_classes, dim]。可以理解为所有item embedding矩阵，那时num_classes=所有item的个数
        biases: 待优化变量，[num_classes]。每个item还有自己的bias，与user无关，代表自己的受欢迎程度。
        labels: 正例的item ids，形状是[batch_size,num_true]的正数矩阵。每个元素代表一个用户点击过的一个item id。允许一个用户可以点击过多个item。
        inputs: 输入的[batch_size, dim]矩阵，可以认为是user embedding
        num_sampled：整个batch要采集多少负样本
        num_classes: 在u2i中，可以理解成所有item的个数
        num_true: 一条样本中有几个正例，一般就是1
        subtract_log_q：是否要对匹配度，进行修正
        remove_accidental_hits：如果采样到的某个负例，恰好等于正例，是否要补救
    Output:
        out_logits: [batch_size, num_true + num_sampled]
        out_labels: 与`out_logits`同形状
    """
```


## reference：
1. [双塔召回模型中的logQ矫正](http://nullpointerexception.top/269)