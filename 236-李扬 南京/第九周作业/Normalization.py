import numpy as np
import matplotlib.pyplot as plt

#归一化1 当前值-最小值/最大值-最小值
def Normalization1(x):
    return [float(i) - min(x)/float(max(x) - min(x)) for i in x]

#归一化1 当前值-最小值/最大值-最小值
def Normalization2(x):
    return [float(i) -np.mean(x) / float(max(x) - min(x)) for i in x]

#标准化
def z_score(x):
    s2 = sum([(i - np.mean(x)) ** 2 for i in x])/len(x)
    return [(i - np.mean(x))/s2 for i in x]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1 = []
cs = []
for i in l:
    # print(i)
    c = l.count(i)
    # print(c)
    cs.append(c)
    print(cs)

n = Normalization2(l)
print(n)
z = z_score(l)
print(z)
plt.plot(l, cs)
plt.plot(z, cs)
plt.show()