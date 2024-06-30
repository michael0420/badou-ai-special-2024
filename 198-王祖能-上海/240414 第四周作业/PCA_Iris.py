'''
鸢尾花分类
@zeno wang
'''

import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)  # 150组数据，分3类，每类50组。各组数据标签有4个。return_X_y = True按照data/target赋值
print(x, y)
print(x.shape, y.shape, type(x), type(y))

pca = dp.PCA(n_components=3)  #
reduced_x = pca.fit_transform(x)  # 主成分分析，注意是先执行pca进行维度n_components=2计算，再小写pca.fit_transform降维数据
print(len(reduced_x), reduced_x)  # 150组数据，变为每组数据特征4个降为2个

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_x)):  # dataset中target就是3类，[0,1,2]
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x, red_y, color='red')
plt.scatter(blue_x, blue_y, color='blue')
plt.scatter(green_x, green_y, color='green')
plt.show()

# 以下是将降维至3阶，并绘制空间散点图
# red_x, red_y, red_z = [], [], []
# blue_x, blue_y, blue_z = [], [], []
# green_x, green_y, green_z = [], [], []
#
# for i in range(len(reduced_x)):
#     if y[i] == 0:
#         red_x.append(reduced_x[i][0])
#         red_y.append(reduced_x[i][1])
#         red_z.append(reduced_x[i][2])
#     elif y[i] == 1:
#         blue_x.append(reduced_x[i][0])
#         blue_y.append(reduced_x[i][1])
#         blue_z.append(reduced_x[i][2])
#     else:
#         green_x.append(reduced_x[i][0])
#         green_y.append(reduced_x[i][1])
#         green_z.append(reduced_x[i][2])
# fig = plt.figure()
# ax3 = plt.axes(projection='3d')
# ax3.scatter(red_x, red_y, red_z, color='red')
# ax3.scatter(blue_x, blue_y, blue_z, color='blue')
# ax3.scatter(green_x, green_y, green_z, color='green')
# plt.show()
