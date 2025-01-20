from torch import nn
from torch.utils._contextlib import F


class WordProbGenerator(nn.Module):
    """
    文本生成器，即把Decoder层的输出通过最后一层softmax层变化为词概率
    """

    def __init__(self, d_model, vocab_size):
        """
        :param d_model: 词向量维度
        :param vocab_size: 词典大小
        """
        super(WordProbGenerator, self).__init__()
        # 通过线性层的映射，映射成词典大小的维度
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # 通过softmax函数对词概率做出估计
        return F.log_softmax(self.linear(x), dim=-1)
