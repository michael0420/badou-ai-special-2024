import numpy as np
import matplotlib.pyplot as plt


# 归一化至0到1的范围，使用(x−x_min)/(x_max−x_min)
def Normalization1(x):

    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]

# 归一化至-1到1的范围，使用(x−x_mean)/(x_max−x_min)
def Normalization2(x):

    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]


# 标准化 x∗=(x−μ)/σ
def z_score(x):

    x_mean = np.mean(x)
    # 计算标准差
    s2 = np.std(x)
    return [(i - x_mean) / s2 for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1 = []

cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
n = Normalization2(l)
z = z_score(l)
print(n)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l, cs)
plt.plot(z, cs)
plt.show()
