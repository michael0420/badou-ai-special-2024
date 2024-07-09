import numpy as np
import matplotlib.pyplot as plt

def Normalization1(x):
    '''归一化（0~1）'''
    '''x_=(x−x_min)/(x_max−x_min)'''
    return [float((i - min(x)) / (max(x) - min(x))) for i in x]

def Normolization2(x):
    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''
    avg = np.mean(x)
    return [float((i - avg) / (max(x) - min(x)))for i in x]
def z_score(x):
    '''x =(x−μ)/σ'''
    avg = np.mean(x)
    sigma = sum([float((i - avg) * (i - avg)) for i in x])/len(x)
    #标准差
    return [(i-avg)/sigma for i in x]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

cs = [i for i in l]
n1 = Normalization1(l)
n2 = Normolization2(l)
z1 = z_score(l)
print(n1)
print(n2)
print(z1)
plt.plot(l, label='orign',lw=2)
plt.plot(n1, label='0~1')
plt.plot(n2, label='-1~1')
plt.plot(z1, label='x =(x−μ)/σ')

plt.legend(loc='best')


plt.show()