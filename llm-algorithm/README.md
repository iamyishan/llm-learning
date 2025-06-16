# LLM算法学习项目

这个仓库包含了大语言模型(LLM)相关算法的学习代码和笔记，从基础的Transformer架构到高级的多模态模型实现。

## 项目结构

### Chapter 01: Transformer基础
- **手动实现Transformer组件**：
  - 自注意力机制
  - 多头注意力机制
  - 因果语言模型解码器
  - LoRA低秩适应技术
- **基础结构实现**：
  - LayerNorm和残差连接
  - 前馈神经网络
- **Transformers库使用示例**：
  - LLaMA模型调试

### Chapter 02: 语言大模型(LLM)
- **DeepSeek模型分析**：
  - 基于Transformer的解码器架构
  - RoPE位置编码
  - 优化的多头注意力实现
  - RMSNorm归一化方法
- **GPT2模型实现**

### Chapter 03: 模型微调
- 手动实现LoRA
- 生成模型的微调技术
- 偏好调整(PPO/DPO)

### Chapter 04: 多模态模型
- MNIST-CLIP图像编码器
- PyTorch扩散模型实现
  - 模型训练
  - LoRA微调

## 环境要求

```bash
torch
transformers
datasets
peft
bitsandbytes
```
## 参考：https://bruceyuan.com/hands-on-code/hands-on-causallm-decoder.html