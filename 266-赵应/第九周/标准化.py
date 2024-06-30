import matplotlib.pyplot as plt
import numpy as np


def max_min_scale(data):
    """最小-最大标准化。将数据映射到[0,1]"""
    max = np.max(data)
    min = np.min(data)
    scale_data = [(x - min) / float(max - min) for x in data]
    return scale_data


def normalization(data):
    mean = np.mean(data)
    max = np.max(data)
    min = np.min(data)
    scale_data = [(x - mean) / (max - min) for x in data]
    return scale_data


def z_score_standardization(data):
    """将数据转换为标准正太分布"""
    mean = np.mean(data)
    squared_diff = [(x - mean) ** 2 for x in data]
    std = sum(squared_diff) / len(data)
    scale_data = [(x - mean) / std for x in data]
    return scale_data


if __name__ == '__main__':
    data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
            11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    result = max_min_scale(data)
    result1 = z_score_standardization(data)
    result2 = normalization(data)
    cs = []
    for i in data:
        c = data.count(i)
        cs.append(c)
    print(cs)
    plt.plot(data, cs)
    # plt.plot(result, cs)
    plt.plot(result1, cs)
    plt.plot(result2, cs)
    plt.show()
