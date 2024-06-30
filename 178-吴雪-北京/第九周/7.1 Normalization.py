"""
标准化（Normalization）
"""
import numpy as np
import matplotlib.pyplot as plt


def Normalization1(x):
    """归一化（0~1）
    x_=(x-x_min)/(x_max-x_min)"""
    return [float(i - min(x)) / float(max(x) - min(x)) for i in x]


def Normalization2(x):
    """归一化（-1~1）
    x_ = (x - x_mean)/(x_max - x_min)"""
    return [float(i - np.mean(x)) / float(max(x) - min(x)) for i in x]


def z_score(x):
    """y = (x - μ) / σ"""
    x_mean = np.mean(x)
    s2 = sum([(i - x_mean) * (i - x_mean) for i in x]) / len(x)
    return [(i - x_mean) / s2 for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

y = []
for i in l:
    c = l.count(i)
    y.append(c)
print('y：\n', y)

f = Normalization1(l)
n = Normalization2(l)
z = z_score(l)
print('f:\n', f)

plt.plot(l, y, c='b')
plt.plot(f, y, c='#E4F300')
plt.plot(n, y, c='g')
plt.plot(z, y, c='r')
plt.legend(labels=['l', 'f', 'n', 'z'])
plt.show()
