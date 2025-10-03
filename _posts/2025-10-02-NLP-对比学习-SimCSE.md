---
title: NLP-对比学习-SimCSE
tags: ["对比学习", "SimCSE", "概念"]
article_header:
  type: cover
  image:
    src: 
---





## 文章来源

SimCSE: Simple Contrastive Learning of Sentence Embeddings

[GitHub - princeton-nlp/SimCSE: [EMNLP 2021\] SimCSE: Simple Contrastive Learning of Sentence Embeddings https://arxiv.org/abs/2104.08821](https://github.com/princeton-nlp/SimCSE)





## Introduction

Our unsupervised SimCSE simply predicts the input sentence itself with only dropout (Srivastava et al., 2014) used as noise (Figure 1(a)). In other words, **we pass the same sentence to the pre-trained encoder twice: by applying the standard dropout twice, we can obtain two different embeddings as “positive pairs”.** **Then we take other sentences in the same mini-batch as “negatives”, and the model predicts the positive one among the negatives.** Although it may appear strikingly simple, this approach outperforms training objectives *such as predicting next sentences (Logeswaran and Lee, 2018) 已被证明没用* and discrete data augmentation (e.g., word deletion and replacement) by a large margin, and even matches previous supervised methods.



unlike previous work that casts it as a 3-way classification task (entailment, neutral, and contradiction), **we leverage the fact that entailment pairs can be naturally used as positive instances.**

- **“3-way classification task” (三向分类任务)**： 这是NLI问题的经典设定。给定一个“前提”和一个“假设”，模型需要判断它们之间的关系是以下三者之一：
  - **Entailment (蕴含)**： 如果前提为真，那么假设必然为真。
    - 前提：猫在垫子上睡觉。
    - 假设：垫子上有一只动物。
  - **Contradiction (矛盾)**： 如果前提为真，那么假设必然为假。
    - 前提：猫在垫子上睡觉。
    - 假设：垫子是空的。
  - **Neutral (中立)**： 前提既不蕴含也不与假设矛盾。
    - 前提：猫在垫子上睡觉。
    - 假设：今天天气很好。
- **“we leverage the fact...” (我们利用了一个事实...)**： 这是本文工作的核心。作者认为，仅仅把“蕴含”当作三个平等标签中的一个，没有充分利用其内在价值。

利用“蕴含对”作为正样本，其背后的逻辑非常强大：

1. **高质量的信号**： “蕴含”关系提供了最清晰、最明确的语义关联信号。如果句子A蕴含句子B，那么它们在语义上必然是高度一致和相关的。这为模型学习“语义相似性”或“语义一致性”提供了完美的监督信号。
2. **构建语义空间**： 通过让模型学习将“蕴含对”中的两个句子的表示拉近，模型可以学会一个更优的**语义表示空间**。在这个空间里，语义相近的句子会离得更近。
3. **可能的新任务形式**： 这种方法通常意味着作者不再做传统的三向分类。他们可能在做：
   - **对比学习**： 将“蕴含对”作为正样本，将“矛盾对”作为硬负样本，来训练模型。
   - **句子编码**： 目标是训练一个句子编码器，使得蕴含的句子对有更高的余弦相似度得分。
   - **二元分类/排序**： 将任务简化为判断一个句子对是否是“蕴含”关系，或者是给句子对的语义相关性打分。







## Background: Contrastive Learning

Contrastive learning aims to learn effective representation **by pulling semantically close neighbors together and pushing apart non-neighbors**

已有研究在多种语境下探索了NLP中类似的对比学习目标（Henderson et al., 2017; Gillick et al., 2019; Karpukhin et al., 2020）。在这些研究中，正样本对 $(x_i, x_i^{'})$ 从有监督数据集中收集，**例如问答-段落对**。由于 $x_i$ 和 $x_i^+$ 在本质上的差异性，这些方法通常采用**双编码器框架**，即为 $x_i$ 和 $x_i^+$ 使用**两个独立的编码器** $f_{\theta_1}$ 和 $f_{\theta_2}$。对于句子嵌入任务，Logeswaran and Lee (2018) 同样采用了**基于双编码器的对比学习**，其通过将**当前句子与下一句**构造成正样本对 $(x_i, x_i^{'})$ 来进行训练。

### 如何度量文本嵌入表示的质量：Alignment and Uniformity

其核心思想是：**正样本的嵌入应当彼此靠近，而随机实例的嵌入应当均匀地分散在超球面上。**

- **目标**：学习一个编码器 $f(x)$，将数据映射到一个单位超球面上。
- **理想效果**：
  1. **相似样本的表示要“靠近”** -> 通过 **Alignment**（对齐性）衡量。
  2. **所有样本的表示要“均匀”地分散在球面上** -> 通过 **Uniformity**（均匀性）衡量。







## Methods： Unsupervised SimCSE

给定一个包含 $N$ 个句子的迷你批次 ${x_i}_{i=1}^N$，对于其中的每个句子 $x_i$，我们将其作为自身的正样本，即 $x_i^+ = x_i$。

#### 核心机制

将同一个句子 $x_i$ 两次输入编码器 $f_\theta$。由于 Transformer 编码器中的 Dropout 机制会在每次前向传播时随机生成掩码，这两次输入会使用两组**独立且不同**的 Dropout 掩码 $z_i$ 和 $z_i'$，从而得到两个在向量空间上略有差异的嵌入表示：

- $h_i^z = f_\theta(x_i, z_i)$
- $h_i^{z'} = f_\theta(x_i, z_i')$

#### 训练目标

对于一个锚点嵌入 $h_i^z$，我们希望在与批次内所有句子的嵌入对比后，能够正确识别出它的正样本 $h_i^{z'}$。

因此，对于迷你批次中的第 $i$ 个句子，其损失函数 $\mathcal{L}_i$ 定义为：
$$
\mathcal{L}_i = -\log \frac{e^{\text{sim}(\mathbf{h}_i^z, \mathbf{h}_i^{z'}) / \tau}}{\sum_{j=1}^N e^{\text{sim}(\mathbf{h}_i^z, \mathbf{h}_j^{z'}) / \tau}}
$$


其中：

- $\text{sim}(h_i, h_j)$ 是余弦相似度函数，即 $ \frac{h_i^T h_j}{|h_i| |h_j|} $
- $\tau$ 是一个温度超参数，用于调节损失函数的敏感度
- 分母中的求和项 $\sum_{j=1}^N$ 涵盖了整个迷你批次的所有 $N$ 个句子。其中，当 $j = i$ 时，分子是分母的一部分，代表正样本；当 $j \neq i$ 时，$h_j^{z'}$ 被视为负样本

最终，整个迷你批次的总体损失是所有 $N$ 个句子损失的平均值：
$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_i
$$

#### 关键要点总结

1. **正样本构造**：正样本来自**同一个句子**，通过**独立的 Dropout 掩码**产生细微差异。
2. **Dropout 的作用**：Dropout 在这里充当了一种高效的、隐式的数据增强手段，它通过随机屏蔽神经元来为正样本对注入噪声。如果没有这种噪声（即使用相同的 Dropout 掩码），$h_i^z$ 和 $h_i^{z'}$ 将完全一致，模型会学到退化的平凡解，导致失败。
3. **对比学习**：该目标是一个标准的对比学习损失（如 InfoNCE），它通过拉近正样本对、推远负样本对（批次内的其他所有句子）来学习高质量的句子嵌入。
4. **无额外操作**：该方法巧妙地利用了 Transformer 固有的 Dropout 机制，无需任何额外的数据增强或网络结构修改，实现了"简单但高效"的设计。



### Experiments

对比不同的data augmentaion方案：包括 Crop k%: keep 100-k% of the length; word deletion k%: delete k% words; Synonym replacement: use nlpaug (Ma, 2019) to randomly replace one word with its synonym; MLM k%: use BERTbase to replace k% of words.

We note that even deleting one word would hurt performance and none of the discrete augmentations outperforms dropout noise

It also demonstrates that starting from a pretrained checkpoint is crucial, for it provides good initial alignment.



We also compare this self-prediction training objective to the next-sentence objective used in Logeswaran and Lee (2018), taking either one encoder or two independent encoders. As shown in Table 2, we find that SimCSE performs much better than the next-sentence objectives (82.5 vs 67.4 on STSB) and **using one encoder instead of two makes a significant difference in our approach**



### Why does it work?

#### 1. **Dropout率实验（Table 3）**

- **最佳设置**：默认的dropout概率（p=0.1）效果最好。
- **其他dropout率**：效果都不如p=0.1。

#### 2. **两个极端案例（关键对比）**

为了理解dropout的作用，作者设计了两个极端对照组：

- **a. “无dropout” (p = 0)**
- **b. “固定0.1” (使用p=0.1，但对句子对使用相同的dropout掩码)**
- **这两者的共同问题**：对于同一个句子，两次前向传播生成的嵌入表示**完全相同**。
- **实验结果**：这两种情况的性能都**急剧下降**。

**为什么性能会急剧下降？**
因为模型失去了学习的目标。如果正样本对的两个嵌入完全一样，那么模型很容易就能让它们的相似度达到最高（1.0），但这学不到任何有意义的、具有鲁棒性的表示。这会导致**模型退化**。

#### 3. **训练过程可视化（Figure 2）**

作者通过两个指标来观察训练过程：

- **对齐性**：正样本对之间的嵌入应该很接近。
- **均匀性**：所有句子的嵌入应该在表示空间中均匀分布，避免坍塌到一个点上。

从图中可以看到：

- **所有模型**：都改善了**均匀性**（因为所有句子嵌入不再挤在一起）。
- **两个极端案例 (p=0 和 固定0.1)**：它们的**对齐性在训练过程中急剧恶化**。
- **无监督SimCSE (使用dropout噪声)**：成功地在**保持良好对齐性**的同时，**也改善了均匀性**。

**结论**：Dropout噪声的作用是**在正样本对中创造可控的、细微的差异**，迫使模型去学习那些**对微小扰动不变的、更本质的语义特征**，从而在保持对齐性的同时优化表示空间。

#### 4. **其他重要发现**

- **预训练模型的重要性**：预训练模型提供了一个**具有良好初始对齐性**的起点。如果没有这个基础，直接从随机初始化开始训练，效果会差很多。
- **与“删除一个词”的对比**：
  - “删除一个词”是一种传统的数据增强方法。
  - 它能**改善对齐性**（因为两个句子的语义几乎没变），但在**改善均匀性**上效果较差。
  - 最终，它的整体性能**不如**基于dropout的SimCSE。

## Methods: Supervised SimCSE

### 核心思想

该方法的核心在于：**利用自然语言推理数据集中的“矛盾”标签，为每个训练样本自动构建一个语义上难以区分的“困难负样本”**，从而让模型学习到更精细的语义表示。

------

### 方法详解

#### 1. **数据来源：NLI 数据集**

- NLI数据集中的每个样本由一个**前提** 和三个由人工标注的假设句组成：
  - **蕴含**：假设句**绝对**为真。
  - **中立**：假设句**可能**为真。
  - **矛盾**：假设句**绝对**为假。
- 文中方法利用了其中的**前提**、**蕴含句** 和**矛盾句**。

#### 2. **构建训练三元组**

- 原始的 `(xi, x+i)` 正样本对被扩展为 `(xi, x+i, x-i)` 三元组。
  - **`xi`**：前提。
  - **`x+i`**：蕴含句，作为**正样本**。
  - **`x-i`**：矛盾句，作为**困难负样本**。

#### 3. **改进的损失函数**

损失函数从对比正样本与批次内所有其他样本，升级为**同时对比正样本和困难负样本**。

**原始损失函数（无监督）:**

text
$$
\mathcal{L}_i = -\log \frac{e^{\text{sim}(\mathbf{h}_i^z, \mathbf{h}_i^{z'}) / \tau}}{\sum_{j=1}^N e^{\text{sim}(\mathbf{h}_i^z, \mathbf{h}_j^{z'}) / \tau}}
$$




**改进的损失函数（有监督，含困难负样本）:**

text
$$
\mathcal{L}_i = -\log \frac{e^{\text{sim}(\mathbf{h}_i, \mathbf{h}_i^+) / \tau}}{\sum_{j=1}^N \left( e^{\text{sim}(\mathbf{h}_i, \mathbf{h}_j^+) / \tau} + e^{\text{sim}(\mathbf{h}_i, \mathbf{h}_j^-) / \tau} \right)}
$$




**关键区别：**

- **分母**不再仅仅包含批次内其他句子的嵌入，而是**明确地加入了每个句子对应的困难负样本的嵌入**。
- 这意味着，模型不仅要学会将锚点 `hi` 拉近其正样本 `h+i`，还要**同时**将其推远一个明确的、具有挑战性的负样本 `h-i`。

------

### 实验结果与发现

1. **性能显著提升**：通过添加困难负样本，模型性能从 **84.9** 提升至 **86.2**，这是最终的有监督SimCSE模型。
2. **其他尝试（未成功）**：
   - **加入ANLI数据集**或**与无监督SimCSE结合**：没有带来有意义的提升。
   - **使用双编码器框架**：性能反而下降（86.2 → 84.2）。这表明在监督学习中，**共享参数的编码器**（即 premise 和 hypothesis 使用同一个编码器）比两个独立的编码器效果更好，因为它能更好地在统一空间中对齐语义。



### **加权负样本**的扩展实验

作者提出了一个直觉想法：**不同类型的负样本应该有不同的重要性**。具体来说，与锚点句子直接相关的“困难负样本”（矛盾句）的重要性，应该与批次中随机其他句子的负样本有所不同,如下公式。**虽然直觉上认为加权可能有帮助，但通过实验证伪了这个假设**
$$
\mathcal{L}_i = -\log \frac{e^{\text{sim}(\mathbf{h}_i, \mathbf{h}_i^+) / \tau}}{\sum_{j=1}^N \left( e^{\text{sim}(\mathbf{h}_i, \mathbf{h}_j^+) / \tau} + \alpha \cdot \mathbb{1}^i_j \cdot e^{\text{sim}(\mathbf{h}_i, \mathbf{h}_j^-) / \tau} \right)}
$$


- **$\alpha$**：**权重超参数**，专门用于控制“困难负样本”的权重。
- **$\mathbb{1}^i_j$**：**指示函数**，这是一个关键设计：
  - 当 $i = j$ 时，$\mathbb{1}^i_j = 1$
  - 当 $i \neq j$ 时，$\mathbb{1}^i_j = 0$

这意味着：

- **只有与当前锚点 $i$ 直接配对的困难负样本 $j=i$ 才会被加权**
- 批次中其他样本 $j \neq i$ 的困难负样本不被特殊加权



## 总结

换到具体场景下 如何构造不同难度的样本？并进一步渐进式学习？
