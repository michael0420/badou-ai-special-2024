import numpy as np
import matplotlib.pylab as plt


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]


def normalization(x):
    return [float(i - min(x)) / float(max(x) - min(x)) for i in x]


def z_score(x):
    x_mean = np.mean(x)
    s2 = sum([(i - x_mean) * (i - x_mean) for i in x]) / len(x)
    return [(i - x_mean) / s2 for i in x]


cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(l)
print(cs)
n = normalization(l)
z = z_score(l)
print(n)
print(z)

plt.plot(l, cs)
plt.plot(n, cs)
plt.plot(z, cs)
plt.show()