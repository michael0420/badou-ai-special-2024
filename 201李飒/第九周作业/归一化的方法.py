import numpy as np
import matplotlib.pyplot as plt
# 归一化的两种方法

def Normalization1(x):
    '''归一化（0~1）
    x= x -x_min / x_max - x_min'''
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]
def Normalization2(x):
    '''归一化（-1~1）
    x= x -x_mean / x_max - x_mean'''
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]
# 标准化
def Z_score(x):
    """x=(x−μ)/σ  μ是均值   σ是标准差（方差开平方根） """
    x_mean = np.mean(x)
    s2 = sum([(i-x_mean)*(i-x_mean) for i in x])/len(x)
    s3 = np.sqrt(s2)
    return [(i-x_mean)/s3 for i in x]




l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10 ]
ln = Normalization1(l)
zn = Normalization2(l)
sn = Z_score(l)
print(ln)
print(zn)
print(sn)

# count 函数 查找列表中元素的数量
cs=[]
for i in l:
    c=l.count(i)
    cs.append(c)
print("........",cs)

plt.plot(l,cs)
plt.plot(ln,cs)
plt.plot(zn,cs)
plt.plot(sn,cs)
plt.show()


