import numpy as np
import matplotlib.pyplot as plt

def normalization1(x):
    """最小-最大归一化"""
    x_min = min(x)
    x_max = max(x)
    return [(float(i) - x_min) / (x_max - x_min) for i in x]

def normalization2(x):
    """均值归一化"""
    x_mean = np.mean(x)
    x_min = min(x)
    x_max = max(x)
    return [(float(i) - x_mean) / (x_max - x_min) for i in x]

def z_score(x):
    """Z-score 标准化"""
    x_mean = np.mean(x)
    x_std = np.std(x)
    return [(i - x_mean) / x_std for i in x]

# 数据列表
l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

# 计算每个数据的出现次数
cs = [l.count(i) for i in l]
print("出现次数:", cs)

# 对数据进行归一化和标准化处理
n = normalization2(l)  # 均值归一化
z = z_score(l)  # Z-score 标准化
print("均值归一化后的数据:", n)
print("标准化后的数据:", z)

# 绘制原始数据和处理后的数据
plt.plot(l, cs, label='Original')
plt.plot(z, cs, label='Z-score')
plt.legend()
plt.show()
