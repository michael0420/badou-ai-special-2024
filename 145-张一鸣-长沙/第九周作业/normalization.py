# coding = utf-8

'''
        实现数据归一化、零均值化
'''

import numpy as np
import matplotlib.pyplot as plt

X = [-10,
     5, 5,
     6, 6, 6,
     7, 7, 7, 7,
     8, 8, 8, 8, 8,
     9, 9, 9, 9, 9, 9,
     10, 10, 10, 10, 10, 10, 10,
     11, 11, 11, 11, 11, 11,
     12, 12, 12, 12, 12,
     13, 13, 13, 13,
     14, 14, 14,
     15, 15,
     30]


def nomalization_1(x):
    # (x-min)/(max-min)
    return [(float((i) - min(x))) / (float(max(x) - min(x))) for i in x]


def normalization_2(x):
    # (x-mean)/(max-min)
    return [(float(i) - np.mean(x)) / (float(max(x)) - min(x)) for i in x]


def zero_mean_normalization(x):
    # (x-μ)/σ   μ：均值，σ：标准差
    sigma = sum([(float(i) - np.mean(x)) * (float(i) - np.mean(x)) for i in x]) / len(x)
    return [(float(i) - np.mean(x)) / sigma for i in x]


n1 = nomalization_1(X)
n2 = normalization_2(X)
zmn = zero_mean_normalization(X)

print(n1)
print(n2)
print(zmn)

zz = []
for i in X:
    num = X.count(i)
    zz.append(num)
print(zz)

plt.plot(X, zz, color="blue")
plt.plot(n1, zz, color="black")
plt.plot(n2, zz, color="purple")
plt.plot(zmn, zz, color="red")
plt.show()
