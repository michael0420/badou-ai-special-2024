import numpy as np
import matplotlib.pyplot as plt

# 归一化两种方式
def Nomalization1(x):
    '''归一化（0，1）
    x_=(x-min)/(max-min)'''
    return [(float(i)-min(x))/(max(x)-min(x)) for i in x]
def Nomalization2(x):
    '''归一化（-1，1）
    x_=2*(x-x_mean)/(max-min)'''
    return [2*(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]
# 标准化
def z_score(x):
    '''x_=(x-μ)/σ'''
    x_mean = np.mean(x)
    s = sum([(float(i)-x_mean)**2 for i in x]) /len(x)
    s2 = s**0.5
    return [(float(i)-x_mean)/s2 for i in x]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)

N = Nomalization2(l)
Z = z_score(l)
print(N)
print(Z)

plt.plot(l,cs)
plt.plot(Z,cs)
plt.show()



