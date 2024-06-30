# coding = utf-8

'''
        实现K-Means聚类算法
'''

from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建数据集
X = [
    [1.748, 0.987],
    [6.8932, 2.833],
    [2.873, 1.832],
    [3.891, 9.983],
    [4.313, 5.232],
    [3.489, 5.8918],
    [3.408, 8.792],
    [8.341, 0.893],
    [8.432, 5.323],
    [5.434, 7.4234]
]
# 调用接口实现K-Means
k = KMeans(n_clusters=4)    # 定义需要聚类的类簇数
res = k.fit_predict(X)      # 得出K-Means结果
print(type(k), k)
print(type(res), res)

# 绘图查看结果
# 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [n[0] for n in X]
y = [n[1] for n in X]
print(x)
print(y)
# 绘制散点图，参数：x横轴; y纵轴; c聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
plt.scatter(x, y, c=res, marker='o', label='聚类')
# 绘图标题
plt.title = 'K-Means'
# 绘制坐标
plt.xlabel('score of science')
plt.ylabel('score of sports')
# 绘制图例
plt.rcParams['font.sans-serif'] = ['simHei']    # 显示中文
plt.legend(loc='best')
plt.show()
