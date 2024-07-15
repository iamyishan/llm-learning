import torch
from torch import nn
import math
from config import *


class TimePositionEmbedding(nn.Module):
    """对时间t进行embedding"""

    def __init__(self, emb_size):
        super().__init__()
        self.half_emb_size = emb_size // 2  # 4
        half_emb = torch.exp(torch.arange(self.half_emb_size) * (-1 * math.log(10000) / (
                self.half_emb_size - 1)))  # [0*(-log(10000)/3), 1*(-log(10000)/3), 2*(-log(10000)/3), 3*(-log(10000)/3)]
        self.register_buffer('half_emb', half_emb)  # 注册一个buffer，不会进行梯度计算,

    def forward(self, t):
        t = t.view(t.size(0), 1)
        half_emb = self.half_emb.unsqueeze(0).expand(t.size(0), self.half_emb_size)
        half_emb_t = half_emb * t
        embs_t = torch.cat((half_emb_t.sin(), half_emb_t.cos()), dim=-1)
        return embs_t


if __name__ == '__main__':
    time_pos_emb = TimePositionEmbedding(8).to(DEVICE)
    t = torch.randint(0, T, (2,)).to(DEVICE)  # 随机2个图片的time时刻
    embs_t = time_pos_emb(t)  # (2,)
    print(embs_t)
