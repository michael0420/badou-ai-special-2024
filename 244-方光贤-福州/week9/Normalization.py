import numpy as np
import matplotlib.pyplot as plt

# 归一化在范围0~1之间
def normalization1(x):
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]

# 归一化在范围-1~1之间
def normalization2(x):
    return [(float(i) - np.mean(x)) / float(max(x) - min(x)) for i in x]

# 标准化
def standardize(x):
    x_mean = np.mean(x)
    s = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    s1 = np.sqrt(s)
    return [(i - x_mean) / s1 for i in x]

l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
n = normalization2(l)
z = standardize(l)
print(n)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l,cs)
plt.plot(z,cs)
plt.show()


