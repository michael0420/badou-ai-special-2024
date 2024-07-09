"""
标准化
"""

import numpy as np
import matplotlib.pyplot as plt

# 0~1
def Normalization1(x):
	return [(float(i)-min(x))/(max(x)-min(x)) for i in x]

# -1~1
def Normalization2(x):
	return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]

#标准化
def z_score(x):
	x_mean = np.mean(x)
	s2 = sum([(float(i) - x_mean)**2 for i in x])/len(x)
	return [(float(i) - x_mean)/s2 for i in x]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs = []
for i in l:
	c = l.count(i)
	cs.append(c)

n = Normalization2(l)
z = z_score(l)

plt.plot(l, cs)
plt.plot(z, cs)
plt.show()