'''实现标准化'''
# 对数据进行归一化处理

import numpy as np
import matplotlib.pyplot as plt


def standardization1(data):
    # 使用y = (x - min) / (max - min)对数据进行归一化(0,1)
    data = np.array(data)

    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data


def standardization2(data):
    # 使用y = (x - mean) / (max - min)对数据进行归一化(-1,1)
    data = np.array(data)

    data = (data - np.mean(data)) / (np.max(data) - np.min(data))
    return data


def z_score(data):
    # 使用z-score对数据进行标准化，y = (x - E) / S
    # E:样本元素均值，S:样本标准差
    data = np.array(data)

    E = np.mean(data)
    S = np.std(data)
    data = (data - E) / S
    return data


if __name__ == '__main__':
    a = np.arange(24).reshape(4,6)
    a = standardization1(a)
    l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
         11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    l1 = []
    cs = []
    for i in l:
        c = l.count(i)
        cs.append(c)
    print(cs)
    n = standardization2(l)
    z = z_score(l)
    print(n)
    print(z)
    '''
    蓝线为原始数据，橙线为z
    '''
    plt.plot(l, cs)
    plt.plot(z, cs)
    plt.show()

    print(a)
