import numpy as np
import matplotlib.pyplot as plt
def normalization1(x):
    # -1-1均值化
    return [ (float(i)-np.mean(x))/ (max(x)-min(x))  for i in x]
def normalization2(x):
    #0-1 均值化
    return [(float(i)-min(x))/ (max(x)-min(x))  for i in x]

def z_normalization(x):
    # x=x-u / sigma
    x_mean = np.mean(x)
    sigma = sum([(i-x_mean)*(i-x_mean) for i in x])/(len(x))
    return [(i-x_mean)/sigma for i in x]


x=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]


res = normalization1(x)
print(res)
cs = []
for i in x:
    c = x.count(i)
    cs.append(c)
print(cs)
n = normalization1(x)
n_z = z_normalization(x)

print(n)
print(n_z)


plt.plot(n,cs)
plt.plot(n_z,cs)
plt.show()