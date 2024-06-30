import numpy as np
import matplotlib.pyplot as plt

def Normalization1(x):
    '''0~1标准化  x_ = (x-x_min)/(x_max-x_min)'''
    x_ = [(float(i)-min(x))/float(max(x)-min(x)) for i in x]
    return x_

def Normalization2(x):
    '''-1~1之间标准化  x_ = (x-u)/(x_max-x_min)'''
    x_ = [(float(i)-np.mean(x))/float(max(x)-min(x)) for i in x]
    return x_
def z_score(x):
    '''0均值化  x_ = (x-u)/σ'''
    u = np.mean(x)
    s = np.std(x)
    x_ = [(i-u)/s for i in x]
    return x_

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
c = []
for i in l:
    c.append(l.count(i))
n1 = Normalization1(l)
n2 = Normalization2(l)
z = z_score(l)
print(l)
print(n1)
print(n2)
print(z)
plt.plot(l,c)
plt.plot(n1,c)
plt.plot(n2,c)
plt.plot(z,c)
plt.show()