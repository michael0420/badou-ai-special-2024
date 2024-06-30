import numpy as np
import matplotlib.pyplot as plt


def normalization1(x):
    # (x-min)/(max-min)归一化结果[0,1]
    return [(i - min(x)) / (max(x) - min(x)) for i in x]


def normalization2(x):
    # (x-mean)/(max-min)归一化结果[-1/2,1/2]
    return [(i - np.mean(x)) / (max(x) - min(x)) for i in x]


def zScore(x):
    xmean = np.mean(x)
    var2 = sum([(i - xmean) * (i - xmean) for i in x]) / len(x)
    return [(i - xmean) / np.sqrt(var2) for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs = []
for i in l:
    count = l.count(i)
    cs.append(count)
z1 = normalization1(l)
z2 = normalization2(l)
z3 = zScore(l)

plt.plot(l, cs, color='blue')
plt.plot(z1, cs, color='orange')
plt.plot(z2, cs, color='g')
plt.plot(z3, cs, color='r')
plt.show()
