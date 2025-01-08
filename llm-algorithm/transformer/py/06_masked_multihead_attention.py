import numpy as np
import torch


def sequence_mask(size):
    """
    序列掩码，解码器输入数据时掩盖后续词的位置
    :param size: 生成词个数
    :return: 右上角为False，主对角线及左下角为True的bool矩阵
    """
    attn_shape = (1, size, size)
    """
    np.triu：返回函数的上三角矩阵A，k=1得到主对角线向上平移一个距离的对角线，
    即保留右上对角线及其以上的数据，其余置为0，即a_11=0
    """
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return (torch.from_numpy(mask) == 0).cuda()  # 通过==0返回的是bool矩阵，即矩阵元素为bool值


if __name__ == '__main__':
    print(sequence_mask(3))
