"""
@author: 207-xujinlan
numpy 实现数据归一化，min_max归一化和z-score归一化
"""

import numpy as np


def min_max(x):
    """
    最大最小值标准化
    :param x: 待标准化数据
    :return: 标准化后的数据
    """
    x_min = np.min(x)
    x_minmax = np.max(x) - x_min
    return [(float(i) - x_min) / x_minmax for i in x]


def z_score(x):
    """
    零均值归一化
    :param x: 待标准化数据
    :return: 标准化后的数据
    """
    x_mean = np.mean(x)
    s2 = sum((i - x_mean) ** 2 for i in x) / len(x)
    return [(i - x_mean) / s2 for i in x]


if __name__ == '__main__':
    x = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,
         10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12,
         12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    m = min_max(x)
    z = z_score(x)
