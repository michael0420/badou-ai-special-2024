from sklearn.decomposition import PCA
from sklearn.datasets._base import load_iris
from matplotlib import pyplot as plt

# x,y = load_iris(return_X_y=True) #加载数据，x表示数据集中的属性数据data，y表示数据标签target
iris = load_iris()
x = iris.data
y = iris.target
pca = PCA(n_components=2)  #对数据进行降维
newx = pca.fit_transform(x)

red_x0,red_x1 = [],[]
bule_x0,blue_x1 = [],[]
green_x0,green_x1 = [],[]
for i in range(len(newx)):   #按不同类别保存数据
    if y[i] == 0:
        red_x0.append(newx[i,0])
        red_x1.append(newx[i,1])
    elif y[i] == 1:
        bule_x0.append(newx[i,0])
        blue_x1.append(newx[i,1])
    else:
        green_x0.append(newx[i,0])
        green_x1.append(newx[i,1])
plt.scatter(red_x0,red_x1,c='r', marker='x')
plt.scatter(bule_x0,blue_x1,c='b', marker='.')
plt.scatter(green_x0,green_x1,c='g', marker='s')
plt.show()
