"""
通过Python的sklearn库来实现鸢尾花数据降维
sklearn.decomposition
sklearn.datasets._base
"""
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris
# PCA降维
x, y = load_iris(return_X_y=True)
pca = dp.PCA(n_components=2)
reduced = pca.fit_transform(x)
# 画图
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced)):
    if y[i] == 0:
        red_x.append(reduced[i][0])
        red_y.append(reduced[i][1])
    elif y[i] == 1:
        blue_x.append(reduced[i][0])
        blue_y.append(reduced[i][1])
    elif y[i] == 2:
        green_x.append(reduced[i][0])
        green_y.append(reduced[i][1])
plt.scatter(red_x, red_y, c='r', marker='D')
plt.scatter(blue_x, blue_y, c='b', marker='.')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
