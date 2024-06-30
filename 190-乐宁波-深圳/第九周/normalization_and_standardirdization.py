import numpy as np
from matplotlib import pyplot as plt


def normalization1(x):
    max_num = max(x)
    min_num = min(x)
    return [(i - min_num) / (max_num - min_num) for i in x]


def normalization2(x):
    x_mean = np.mean(x)
    return [(i - x_mean) / x_mean for i in x]


def z_score(x):
    x_mean = np.mean(x)
    std_dev = np.std(x)
    return [(i - x_mean) / std_dev for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15,
     30]
l1 = []
# for i in l:
#     i+=2
#     l1.append(i)
# print(l1)
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
n = normalization1(l)
n2 = normalization2(l)
z = z_score(l)
print(n)
print(n2)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l, cs)
plt.plot(z, cs)
plt.show()
