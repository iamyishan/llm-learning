import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    正弦位置编码，即通过三角函数构建位置编码

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    """

    def __init__(self, dim: int, dropout: float, max_len=5000):
        """
        :param dim: 位置向量的向量维度，一般与词向量维度相同，即d_model
        :param dropout: Dropout层的比率
        :param max_len: 句子的最大长度
        """
        # 判断能够构建位置向量
        if dim % 2 != 0:
            raise ValueError(f"不能使用 sin/cos 位置编码，得到了奇数的维度{dim:d}，应该使用偶数维度")

        """
        构建位置编码pe
        pe公式为：
        PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})
        """
        pe = torch.zeros(max_len, dim)  # 初始化pe
        position = torch.arange(0, max_len).unsqueeze(1)  # 构建pos，为句子的长度，相当于pos
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * torch.tensor(
            -(math.log(10000.0) / dim))))  # 复现位置编码sin/cos中的公式
        pe[:, 0::2] = torch.sin(position.float() * div_term)  # 偶数使用sin函数
        pe[:, 1::2] = torch.cos(position.float() * div_term)  # 奇数使用cos函数
        pe = pe.unsqueeze(1)  # 扁平化成一维向量

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)  # pe不是模型的一个参数，通过register_buffer把pe写入内存缓冲区，当做一个内存中的常量
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """
        词向量和位置编码拼接并输出
        :param emb: 词向量序列（FloatTensor），``(seq_len, batch_size, self.dim)``
        :param step: 如果 stepwise("seq_len=1")，则用此位置的编码
        :return: 词向量和位置编码的拼接
        """
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]  # 拼接词向量和位置编码
        else:
            emb = emb + self.pe[step]
        emb = self.drop_out(emb)
        return emb
