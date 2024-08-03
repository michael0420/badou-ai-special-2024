import numpy as np
import matplotlib.pyplot as plt

def Normalization(x):
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]

def z_score(x):
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    return [(i - x_mean) / s2 for i in x]

l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs = [l.count(i) for i in l]
print(cs)

n = Normalization(l)
z = z_score(l)
print(n)
print(z)

plt.plot(l, cs, label='origin data')
plt.plot(n, cs, label='normalization data')
plt.plot(z, cs, label='standardization data')
plt.legend()
plt.xlabel('data value')
plt.ylabel('count')
plt.title('Data Normalization Example')
plt.show()