'''
#调用sklearn库进行PCA算法降维

import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2],[3,5,42,8]])
pca = PCA(n_components=3)    #降维
pca.fit(X)    # 执行X
newX = pca.fit_transform(X)   #降维后的数据
print(newX)
'''

# 鸢尾花示例
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris

x, y = load_iris(return_X_y=True)  #加载数据，x:属性数据， y：数据标签（类型
pca = dp.PCA(n_components=2)    #降维后的主成分数目
New_x = pca.fit_transform(x)   #降维后形成的新矩阵

print(New_x)
print(len(New_x))

red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]

for i in range(len(New_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中（共150个数据）
    if y[i]==0:
        red_x.append(New_x[i][0])
        red_y.append(New_x[i][1])
    elif y[i]==1:
        blue_x.append(New_x[i][0])
        blue_y.append(New_x[i][1])
    else:
        green_x.append(New_x[i][0])
        green_y.append(New_x[i][1])

plt.scatter(red_x,red_y,c='r',marker='x')    #marker:形状
plt.scatter(blue_x,blue_y,c='b',marker='o')
plt.scatter(green_x,green_y,c='g',marker='<')
plt.show()