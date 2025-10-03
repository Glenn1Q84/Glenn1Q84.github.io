---
title: NLP-Albert
tags: ["NLP", "BERT", "ALBERT"]
article_header:
  type: cover
  image:
    src: 
---





l

论文：[[1909.11942\] ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)

代码：https://github.com/google-research/ALBERT

# Introduction



Motivation: **应对GPU/TPU memory limitations and  longer training times。**this study present two parameter reduction techniques to **lower memory consumption** and **increase the training  speed of BERT**

The first one is a **factorized embedding parameterization.** By decomposing  the large vocabulary embedding matrix into two small matrices, we separate the size of the hidden  layers from the size of vocabulary embedding. This separation makes it easier to grow the hidden  size without significantly increasing the parameter size of the vocabulary embeddings. The second  technique is **cross-layer parameter sharing.** This technique prevents the parameter from growing  with the depth of the network. The parameter reduction techniques also act as **a form of regularization that stabilizes the training  and helps with generalization**

To further improve the performance of ALBERT, we also introduce a **self-supervised loss for  sentence-order prediction (SOP)**. SOP primary focuses on inter-sentence coherence and is designed  to **address the ineffectiveness (Yang et al., 2019; Liu et al., 2019) of the next sentence prediction  (NSP) loss** proposed in the original BERT. We compare to this loss in our experiments and find that  **sentence ordering is a more challenging pretraining task and more useful for certain downstream  tasks.** 为什么顺序比 原始的句子对更有挑战



# Methods

### Factorized embedding parameterization

word embedding size: E

hidden-layer embedding size: H

In BERT, as well as subsequent modeling improvements such as XLNet and RoBERTa , the WordPiece embedding  size E is tied with the hidden layer size H, i.e., **E ≡ H.** 

**word embeddings** are meant to learn **context-independent representations**, whereas **hidden-layer embeddings** are meant to learn **context-dependent representations**. 意味着 word embedding是固定的词汇信息，hidden-layer是语境变化的语义信息. As such, untying（release/reduce） the WordPiece embedding size E from the hidden layer size H  allows us to make a more efficient usage of the total model parameters as informed by modeling  needs, which dictate that H>E.

Therefore, for ALBERT we use a **factorization of the embedding parameters, decomposing them  into two smaller matrices.** **Instead of projecting the one-hot vectors directly into the hidden space of  size H, we first project them into a lower dimensional embedding space of size E, and then project  it to the hidden space**. By using this decomposition, we reduce the embedding parameters from  O(V × H) to O(V × E + E × H). This parameter reduction is significant when H>>E. We  choose to use the same E for all word pieces because they are much more evenly distributed across  documents compared to whole-word embedding, where having different embedding size (Grave  et al. (2017); Baevski & Auli (2018); Dai et al. (2019) ) for different words is important.

原来 直接从embedding size 映射到hidden size, 现在通过两次变换：embedding size->smaller size->hidden size

中间这个变换 实现了 语义的压缩到扩充的过程。**类似的，可以用来处理 多个任务之间的并行，或者多种信息的关注，**



###  Cross-layer parameter sharing 

For ALBERT, we propose cross-layer parameter sharing as another way to improve parameter efficiency. There are multiple ways to share parameters, e.g., only  sharing feed-forward network (FFN) parameters across layers, or only sharing attention parameters.  **The default decision for ALBERT is to share all parameters across layers**



### why SOP? Inter-sentence coherence loss

NSP

BERT uses an additional loss called next-sentence prediction (NSP). NSP is a  binary classification loss for **predicting whether two segments appear consecutively in the original  text,** as follows: **positive examples are created by taking consecutive segments from the training  corpus; negative examples are created by pairing segments from different documents;** positive and  negative examples are sampled with equal probability.



分析：We conjecture that the main reason behind NSP’s ineffectiveness **is its lack of difficulty as a task**,  as compared to MLM. As formulated, NSP conflates(merge) *topic prediction and coherence prediction* in a single task2. However, **topic prediction is easier to learn compared to coherence prediction**, and **also  overlaps more with what is learned using the MLM loss**.

We maintain that **inter-sentence modeling is an important aspect of language understanding**, but we  propose a loss **based primarily on coherence.** That is, for ALBERT, we use a sentence-order prediction (SOP) loss, which **avoids topic prediction** and instead **focuses on modeling inter-sentence  coherence**.

The SOP loss uses **as positive examples the same technique as BERT (two consecutive segments from the same document)**, and **as negative examples the same two consecutive segments but with their order swapped**（顺序交换）. This **forces the model to learn finer-grained distinctions about  discourse-level coherence properties**.



| 任务类型  | 正样本构造               | 负样本构造                                         | 核心挑战与动机                                             | 优点                                                         | 缺点                                                         | 使用模型                |
| :-------- | :----------------------- | :------------------------------------------------- | :--------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :---------------------- |
| **NSP**   | 同一段落中连续的两个句子 | 从不同文档随机选取的句子，或同一文档非相邻段落选取 | 学习句子间的连贯性；为下游任务（如问答）做准备             | 实现简单，直观                                               | 任务过于简单；负样本“太负”，模型可能只学会主题识别，而非逻辑理解 | 原始BERT                |
| **SOP**   | 同一段落中连续的两个句子 | **交换**正样本中两个句子的顺序                     | 解决NSP的缺陷；**迫使模型学习更深层的逻辑和时序关系**      | 任务更具挑战性；正负样本主题一致，模型必须关注逻辑顺序       | 实现比NSP稍复杂                                              | ALBERT                  |
| **无NSP** | **不专门构造**           | **不专门构造**                                     | 认为NSP任务低效或有害；**证明充分优化的MLM本身已足够强大** | 简化训练流程；避免NSP任务的噪声；在许多任务上表现优于原始BERT | 移除了一个显式的句子关系预训练任务                           | RoBERTa, 及后续许多模型 |



在BERT的NSP中，通过在不同的documents采样 构成负样本、在相同的document构造正样本，这种方式本质上包含着  主题建模任务，即判断两句话语义上是否相关（相关的来自相同的document,不相关的来自不同的document）

topic prediction 与 coherence重叠，

# Experiments

### Cross-layer parameter sharing

 

Table 4 presents experiments for various cross-layer parameter-sharing strategies, using an  ALBERT-base configuration (Table 1) with two embedding sizes (E = 768 and E = 128). **We  compare the all-shared strategy (ALBERT-style), the not-shared strategy (BERT-style), and intermediate strategies in which only the attention parameters are shared (but not the FNN ones) or only  the FFN parameters are shared (but not the attention ones)**.  **The all-shared strategy hurts performance under both conditions**, but it is less severe for E = 128 (1.5 on Avg) compared to E = 768 (-2.5 on Avg). In addition, **most of the performance drop appears  to come from sharing the FFN-layer parameters, while sharing the attention parameters results in no  drop when E = 128 (+0.1 on Avg), and a slight drop when E = 768 (-0.7 on Avg).**

### Sentence Order Prediction (SOP)

We compare head-to-head three experimental conditions for the additional inter-sentence loss: none  (XLNet- and RoBERTa-style), NSP (BERT-style), and SOP (ALBERT-style), using an ALBERT-base configuration. Results are shown in Table 5, both over intrinsic (accuracy for the MLM, NSP,  and SOP tasks) and downstream tasks.

The results on the intrinsic tasks reveal that **the NSP loss brings no discriminative power to the SOP  task** (52.0% accuracy, similar to the random-guess performance for the “None” condition). This  allows us to conclude that NSP ends up modeling only topic shift. In contrast, the SOP loss does  solve the NSP task relatively well (78.9% accuracy), and the SOP task even better (86.5% accuracy).  Even more importantly, **the SOP loss appears to consistently improve downstream task performance  for multi-sentence encoding tasks** (around +1% for SQuAD1.1, +2% for SQuAD2.0, +1.7% for  RACE), for an Avg score improvement of around +1%.





# 总结与分析

Mask任务与句子对预测任务本质上可以抽象为？

​	Mask，与完形填空类似，能考察 上下文理解，逻辑关系推理，词汇辨析。**这些任务包含了主题预测**

