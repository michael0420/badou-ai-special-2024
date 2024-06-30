"""
标准化
@Author：zsj
"""
import numpy as np
import matplotlib.pyplot as plt


# 归一化0-1  x_=(x−x_min)/(x_max−x_min)
def normalization1(x):
    max_v = max(x)
    min_v = min(x)
    range = max_v - min_v
    if range == 0:
        return x
    return [((i - min_v) / range) for i in x]


# 归一化 -1~1  x_=(x−mean)/(x_max−x_min)
def normalization2(x):
    mean = np.mean(x)
    range = max(x) - min(x)
    if range == 0:
        return x
    return [((i - mean) / range) for i in x]


# z-score标准化 x=(x−μ)/σ
def z_score(x):
    mean = np.mean(x)
    s2 = sum([(i - mean) ** 2 for i in x]) / len(x)
    σ = np.sqrt(s2)
    if σ == 0:
        return x
    return [(i - mean) / σ for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1 = []
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
n = normalization2(l)
z = z_score(l)

print(n)
print(z)
plt.plot(l, cs)
plt.plot(n, cs)
plt.plot(z, cs)
plt.show()
