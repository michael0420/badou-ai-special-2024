from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute
"""
X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
     ]

# 输出数据集
print(X)

"""
第二部分：KMeans聚类
clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
y_pred = clf.fit_predict(X) 载入数据集X，并且将聚类的结果赋值给y_pred
"""

clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(X)

# 输出完整Kmeans函数，包括很多省略参数
print(clf)
# 输出聚类预测结果
print("y_pred = ", y_pred)

"""
第三部分：可视化绘图
"""

import numpy as np
import matplotlib.pyplot as plt

# 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [n[0] for n in X]
print(x)
y = [n[1] for n in X]
print(y)

''' 
绘制散点图 
参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
'''
plt.scatter(x, y, c=y_pred, marker='x')

# 绘制标题
plt.title("Kmeans-Basketball Data")

# 绘制x轴和y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")

# 设置右上角图例
plt.legend(["A", "B", "C"])

# 显示图形
plt.show()


"""
代码实现了使用 scikit-learn 库的 KMeans 类对一组篮球运动员的比赛数据进行聚类分析。下面是代码的简化和解释：

from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
 
# 数据集 - 篮球运动员的比赛数据 
X = [
    [0.0888, 0.5885],
    [0.1399, 0.8291],
    [0.0747, 0.4974],
    # ... 省略其他数据 ...
    [0.1956, 0.4280]
]
 
# 创建 KMeans 对象，设置类簇数为 3 
clf = KMeans(n_clusters=3)
 
# 适合数据并进行预测 
y_pred = clf.fit_predict(X)
 
# 绘制散点图，展示聚类结果 
plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker='x')  # 使用 X 的第一列和第二列数据绘制散点图 
 
# 设置图表标题和坐标轴标签 
plt.title("K-Means Clustering of Basketball Players' Data")
plt.xlabel("Assists Per Minute")
plt.ylabel("Points Per Minute")
 
# 显示图表 
plt.show()

在这个例子中：

    KMeans(n_clusters=3) 创建了一个 KMeans 对象，用于后续的聚类操作。
    fit_predict(X) 方法适合数据集 X 并返回预测的类别标签。
    plt.scatter() 函数根据数据集的第一列和第二列绘制散点图，并用不同颜色表示不同的类标签。
    最后，图表展示了不同类别的球员在助攻和得分方面的分布情况。

请注意，这个简单的例子只考虑了两个特征（每分钟助攻数和每分钟得分数），并且假设数据集已经按需预处理。
在实际应用中，可能需要处理更复杂的数据集，并且可能需要进行特征选择、归一化等预处理步骤。
此外，K-Means 算法的结果可能会因为初始化中心的选择而有所不同，有时候可能需要多次运行并选择最好的结果。
"""