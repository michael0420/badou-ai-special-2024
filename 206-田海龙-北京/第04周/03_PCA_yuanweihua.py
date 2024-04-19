
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

x,y=load_iris(return_X_y=True) #加载数据，x表示数据集中的属性数据，y表示数据标签

# 原始数据
print(x,y.reshape(-1,1))

red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i]==0:
        red_x.append(x[i][0])
        red_y.append(x[i][1])
    elif y[i]==1:
        blue_x.append(x[i][0])
        blue_y.append(x[i][1])
    else:
        green_x.append(x[i][0])
        green_y.append(x[i][1])

plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()

pca=dp.PCA(n_components=2) #加载pca算法，设置降维后主成分数目为2
reduced_x=pca.fit_transform(x) #对原始数据进行降维，保存在reduced_x中

print("最终结果：\n",reduced_x)


red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(reduced_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i]==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

# 展示散点图，可视化数据分布情况，其实就是降维后两个特征的可区分度
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()
