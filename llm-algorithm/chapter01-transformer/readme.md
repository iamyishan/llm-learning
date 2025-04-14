# Chapter01 Transformer 学习笔记

本章节主要学习和实现Transformer模型的核心组件，包括自注意力机制、多头注意力、解码器等，以及一些优化技术如LoRA。

## 目录结构

- `hand-transformer/`: 手动实现Transformer的各个组件
  - `01_hand_self_attention.ipynb`: 自注意力机制的实现
  - `02_hand_mutil_head_attention.ipynb`: 多头注意力机制的实现
  - `03_hand_causal_lm_decoder.ipynb`: 因果语言模型解码器的实现
  - `04_hand_lora.ipynb`: LoRA低秩适应技术的实现
- `transformers/`: 使用Hugging Face Transformers库的示例
  - `01_debug_llama.py`: LLaMA模型的调试示例

## 核心概念

### 自注意力机制 (Self-Attention)

自注意力的核心公式：
$$\text{SelfAttention}(X) = \text{softmax}\left(\frac{Q \cdot K}{\sqrt{d}}\right) \cdot V$$

其中 $Q = W_Q \times X$, $K = W_K \times X$, $V = W_V \times X$，分别通过不同的权重矩阵投影得到。

自注意力机制的实现分为几个层次：
1. 基础实现：直接按照公式实现
2. 效率优化：合并QKV矩阵计算
3. 细节完善：添加dropout、attention mask等

### 多头注意力 (Multi-Head Attention)

多头注意力将输入分割成多个头，每个头独立计算自注意力，然后将结果拼接并通过输出投影矩阵。这种方式允许模型关注不同位置的不同表示子空间信息。

### 因果语言模型解码器 (Causal LM Decoder)

与原始Transformer的解码器相比，CausalLM解码器简化了结构：
- 移除了encoder-decoder交叉注意力层
- 流程简化为：input → self-attention → FFN
- 使用了残差连接和层归一化

主要有两种归一化方式：
- pre-norm: x + Sublayer(LayerNorm(x))
- post-norm: LayerNorm(x + Sublayer(x))

### LoRA (Low-Rank Adaptation)

LoRA是一种参数高效的微调方法，核心原理是通过低秩分解来减少需要训练的参数量：

$$W_{new} = W_0 + AB$$

其中$W_0$是预训练权重（保持冻结），$A$和$B$是小型低秩矩阵（可训练）。

LoRA的优点：
- 节约显存
- 训练速度快
- 效果损失较小
- 推理时不增加耗时

## 补充知识点

- 为什么自注意力计算中要除以$\sqrt{d}$：防止梯度消失，保持QK内积分布与输入一致
- RMSNorm与LayerNorm的区别：RMSNorm只进行缩放而不进行重新中心化
- 预归一化(pre-norm)与后归一化(post-norm)的区别及其对训练稳定性的影响