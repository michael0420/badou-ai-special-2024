#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris

x,y = load_iris(return_X_y = True)
pca = dp.PCA(n_components=2)
reduced_x = pca.fit_transform(x)

#init
r_x,r_y = [],[]
b_x,b_y = [],[]
g_x,g_y = [],[]
for i in range(len(reduced_x)):
	if y[i]==0:
		r_x.append(reduced_x[i][0])
		r_y.append(reduced_x[i][1])
	if y[i]==1:
		b_x.append(reduced_x[i][0])
		b_y.append(reduced_x[i][1])
	else:
		g_x.append(reduced_x[i][0])
		g_y.append(reduced_x[i][1])
plt.scatter(r_x, r_y,c='r',marker='x')
plt.scatter(b_x, b_y,c='b',marker='D')
plt.scatter(g_x,g_y,c='g',marker='.')
plt.show()