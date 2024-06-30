import numpy as np
import matplotlib.pyplot as plt

# 数据归一化处理的两种方式
'''
映射到[0,1]区间上
x = (x-x_min)/(x_max-x_min)
'''
def Normalization1(x):
    return [float(i)-min(x)/float(max(x)-min(x)) for i in x]

'''
映射到[-1,1]区间上
x = (x-x_mean)/(x_max-x_min)
'''
def Normalization2(x):
    return [(float(i)-np.mean(x))/float(max(x)-min(x)) for i in x]



# 零均值归一化（zero-mean normalization）

'''
z-score（标准正态分布）
x = (x-μ)/σ
'''
def z_score(x):
    x_mean = np.mean(x)
    sigma = sum([(i-x_mean)**2 for i in x])/len(x)
    return [(i-x_mean)/sigma for i in x]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1=[]

# 数相同数字有几个
cs=[]
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)

n1 = Normalization1(l)
n2 = Normalization2(l)
z = z_score(l)

print('n1',n1)
print('n2',n2)


plt.plot(l,cs)
plt.plot(n1,cs)
plt.plot(n2,cs)
plt.plot(z,cs)
plt.show()
