import numpy as np
import matplotlib.pyplot as plt
def normalization1(x):
    '''归一化（0~1）'''
    nor = []
    for i in x:
        n = (float(i)-min(x))/float(max(x)-min(x))
        nor.append(n)
    return nor
def normalization2(x):
    '''归一化（-1~1）'''
    nor = []
    for i in x:
        n = (float(i)-np.mean(x))/float(max(x)-min(x))
        nor.append(n)
    return nor
def z_score(x):
    #标准化
    '''x∗=(x−μ)/σ'''
    mean = np.mean(x)
    s = sum([(i-mean)*(i-mean) for i in x])/len(x)
    return [(i-mean)/s for i in x]
l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1=[]
cs=[]
for i in l:
    c=l.count(i) #计算出现次数
    cs.append(c)
print(cs)
n1 = normalization1(l)
n2 = normalization2(l)
n3 = z_score(l)
plt.plot(l,cs)
plt.plot(n1,cs,color='red')
plt.plot(n2,cs,color='green')
plt.plot(n3,cs,color='yellow')
plt.show()