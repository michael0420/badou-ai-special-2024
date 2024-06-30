import numpy as np
import matplotlib.pyplot as plt


# 归一化的两种方式
def Normalization1(x):
    '''归一化（0~1）'''
    '''x_=(x−x_min)/(x_max−x_min)'''
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


def Normalization2(x):
    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]


# 标准化
def z_score(x):
    '''x∗=(x−μ)/σ'''
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    return [(i - x_mean) / s2 for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs = []
for i in l:
    c = l.count(i)  # 计算每个元素在列表 l 中的出现次数
    cs.append(c)
print(cs)
n = Normalization2(l)
z = z_score(l)
print(n)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l, cs)   # 横轴为数组l，纵轴为数组中元素出现的次数
plt.plot(z, cs)
plt.show()
