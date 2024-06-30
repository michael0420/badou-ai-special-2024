#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def Normalization(x):
	return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

def z_score(x):
	x_mean = np.mean(x)
	s = sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
	return [(i-x_mean)/s for i in x]

L = [-10,-9,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,10,10,10,10,10,11,11,11,11,11,13,13,13,13,13,13,15,1,15,15,15]

L1 = []

cs =[]

for i in L:
	c = L.count(i)
	cs.append(c)
	
n = Normalization(L)
z = z_score(L)

plt.plot(L,cs)
plt.plot(z,cs)
plt.show()