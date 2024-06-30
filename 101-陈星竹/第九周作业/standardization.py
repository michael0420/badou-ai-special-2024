import numpy as np
import matplotlib.pyplot as plt

#归一化
#[0,1]
'''
(x-min)/(max-min)
'''
def Normalization1(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

#[-1,1]
'''
(x-mean)/(max-min)
'''
def Normalization12(x):
    return [(float(i)-np.mean(x))/(float(max(x)-min(x))) for i in x]
#标准化
#正态分布
'''
(x-u)/σ
'''
def zero_mean(x):
    u = np.mean(x)
    σ = np.std(x)  # 使用 np.std() 计算标准差
    if σ == 0:
        return [0 for i in x]  # 如果标准差为零，返回全零列表
    return [(i - u) / σ for i in x]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
y = []

#计算纵坐标
for i in l:
    c=l.count(i)
    y.append(c)

n1 = Normalization1(l)
n2=Normalization12(l)
z=zero_mean(l)
print(n1)
print(n2)
print(z)


plt.plot(l,y)
plt.plot(z,y)
plt.show()
