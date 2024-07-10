import numpy as np
import matplotlib.pyplot as plt

'''
蓝线为原始数据，橙线为z
'''
#归一化实现
#原理很简单，举个例子：例如数值范围在0-10，x=5的点占多少比例，就把5/(10-0)=0.5。
def Normalization_0_1(x):
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]

#-1~1归一化，以平均点为分界，减去平均点除以数值范围即可
def Normalization_1_1(x):
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]

#标准化
def z_score(x):
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    return [(i - x_mean) / s2 for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1 = []
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
n = Normalization_0_1(cs)
z = z_score(l)
print(n)
print(z)

plt.plot(l, cs)
plt.plot(n, cs)
plt.show()
