import matplotlib.pyplot as plt
import sklearn.decomposition as de
from sklearn.datasets._base import load_iris


x, y = load_iris(return_X_y=True)
#调用PCA算法进行降维，n_components = x代表降维到x维度
pca =de.PCA(n_components=2)
result =pca.fit_transform(x)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
#按降维后的数据保存在不同数组中，用于进行表示
for i in range(len(result)):
    if y[i] == 0:
        red_x.append(result[i][0])
        red_y.append(result[i][1])
    elif y[i] == 1:
        blue_x.append(result[i][0])
        blue_y.append(result[i][1])
    else:
        green_x.append(result[i][0])
        green_y.append(result[i][1])
plt.scatter(red_x, red_y, c='r', marker='.')
plt.scatter(blue_x, blue_y, c='b', marker='*')
plt.scatter(green_x, green_y, c='g', marker='x')
plt.show()