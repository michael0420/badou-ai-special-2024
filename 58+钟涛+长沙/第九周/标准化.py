import numpy as np
import matplotlib.pyplot as plt

'''归一化（0~1）'''
'''x_=(x−x_min)/(x_max−x_min)'''
def Normalization1(x):
    return [float((i - np.min(x))/(np.max(x) - np.min(x)) for i in x)]

'''归一化（-1~1）'''
'''x_=(x−x_mean)/(x_max−x_min)'''
def Normalization2(x):
    x_mean = np.mean(x)
    return [float((i - x_mean)/(np.max(x) - np.min(x)) for i in x)]

#标准化公式
'''x∗=(x−μ)/σ'''
#μ是均值
#σ 是标准差  (x-μ)^2/n  ∑
def z_score(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_std = sum([float(( i - x_mean) ** 2 / len(x)) for i in x]) ** 0.5
    return [float(i - x_mean) / x_std for i in x];

data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

z = z_score(data)
cs=[]
for i in data:
    c=data.count(i)
    cs.append(c)
print(cs)
print(z)

plt.plot(data,cs)
plt.plot(z,cs)
plt.show()
