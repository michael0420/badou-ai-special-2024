import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
#归一化的两种方式
def Normalization1(x):
    '''归一化（0~1）'''
    '''x_=(x−x_min)/(x_max−x_min)'''
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)
def Normalization2(x):
    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''
    x_mean = np.mean(x)
    x_max = np.max(x)
    x_min = np.min(x)
    return (x - x_mean) / (x_max - x_min)
#标准化
def z_score(x):
    '''x∗=(x−μ)/σ'''
    x_mean = np.mean(x)
    s2 = np.sum((x - x_mean)**2) / len(x)
    return (x - x_mean) / s2
 
l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1=[]
# for i in l:
#     i+=2
#     l1.append(i)
# print(l1)
counts = Counter(l)
cs = [counts[i] for i in l]
print(cs)
n=Normalization2(l)
z=z_score(l)
print(n)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l, cs, label='Original')
plt.plot(n,cs, label='Normalization')
plt.plot(z, cs, label='Z-score')
plt.legend()
plt.show()

