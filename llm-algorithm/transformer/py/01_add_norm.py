import torch
from torch import nn



class LayerNorm(nn.Module):
    """
    构建一个LayerNorm Module
    LayerNorm的作用：对x归一化，使x的均值为0，方差为1
    LayerNorm计算公式：x-mean(x)/\sqrt{var(x)+\epsilon} = x-mean(x)/std(x)+\epsilon
    """

    def __init__(self, x_size, eps=1e-6):
        """
        :param x_size: 特征的维度
        :param eps: eps是一个平滑的过程，取值通常在（10^-4~10^-8 之间）
        其含义是，对于每个参数，随着其更新的总距离增多，其学习速率也随之变慢。
        防止出现除以0的情况。

        nn.Parameter将一个不可训练的类型Tensor转换成可以训练的类型parameter，
        并将这个parameter绑定到这个module里面。
        使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        """
        super(LayerNorm, self).__init__()
        self.ones_tensor = nn.Parameter(torch.ones(x_size))  # 按照特征向量大小返回一个全1的张量，并且转换成可训练的parameter类型
        self.zeros_tensor = nn.Parameter(torch.zeros(x_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)  # 求标准差
        return self.ones_tensor * (x - mean) / (std + self.eps) + self.zeros_tensor  # LayerNorm的计算公式

class SublayerConnection(nn.Module):
    """
    子层的连接: layer_norm(x + sublayer(x))
    上述可以理解为一个残差网络加上一个LayerNorm归一化
    """

    def __init__(self, size, dropout=0.1):
        """
        :param size: d_model
        :param dropout: drop比率
        """
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        # TODO：在SublayerConnection中LayerNorm可以换成nn.BatchNorm2d
        # self.layer_norm = nn.BatchNorm2d()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        :param x: 就是self-attention的输入
        :param sublayer: self-attention层,sublayer(x)是self-attention层的输出
        :return:
        """
        return self.dropout(self.layer_norm(x + sublayer(x)))