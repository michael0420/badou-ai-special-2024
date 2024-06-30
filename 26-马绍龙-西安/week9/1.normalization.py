import numpy as np
import matplotlib.pyplot as plt


# 归一化的两种方式
def Normalization1(x):
    '''归一化（0~1）'''
    '''x_=(x−x_min)/(x_max−x_min)'''
    diff = float(max(x) - min(x))
    return [(float(i) - min(x)) / diff for i in x]


def Normalization2(x):
    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''
    diff = float(max(x) - min(x))
    return [(float(i) - np.mean(x)) / diff for i in x]


# z-score标准化
def z_score(x):
    '''x∗=(x−μ)/σ'''
    x_mean = np.mean(x)
    s2 = np.sqrt(sum([(i - x_mean) * (i - x_mean) for i in x]) / len(x))
    return [(i - x_mean) / s2 for i in x]


input = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
         11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

cs = []
for i in input:
    c = input.count(i)  # 计算元素i在列表l中出现的次数
    cs.append(c)  # 将次数添加到列表cs中
print(cs)

n1 = Normalization1(input)
print(n1)

n2 = Normalization2(input)
print(n2)

z = z_score(input)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(input, cs)  # 参数l表示x轴上的数据点，cs表示每个x值对应的y值
plt.plot(z, cs)
plt.plot(n1, cs)
plt.plot(n2, cs)
plt.show()
