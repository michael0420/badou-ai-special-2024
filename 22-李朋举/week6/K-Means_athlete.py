# coding=utf-8  
from sklearn.cluster import KMeans

"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据，总共20行，每行两列数据
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
clf = KMeans(n_clusters=3) 表示类簇数为3，指定聚成3类数据，clf即赋值为KMeans
y_pred = clf.fit_predict(X) 载入数据集X，并且将聚类的结果赋值给y_pred
 
     fit_predict()是 scikit-learn 库中 KMeans 模型的一个方法，用于对数据进行聚类，并返回聚类结果。
     KMeans 模型的 fit_predict()方法的详细解释如下：
          1. **fit**: 这个方法首先在数据上执行 KMeans 算法的拟合步骤。在拟合过程中，模型会根据指定的参数（如 n_clusters）尝试确定数据的最佳聚类中心。
                      拟合过程通常涉及计算数据点与各个聚类中心之间的距离，并根据距离将数据点分配给最接近的聚类中心。
          2. **predict**: 在拟合步骤之后，predict 方法使用拟合好的模型对新数据进行预测。预测过程涉及计算新数据点与拟合步骤中确定的聚类中心之间的距离，
                          并根据距离将新数据点分配给最接近的聚类中心。预测结果通常以整数形式返回，表示每个数据点所属的聚类中心的索引。
          3.`fit_predict()`是 KMeans 类的一个方法，它的入参和出参如下所示：
                    入参：
                    - X：数值型矩阵，维度为(n_samples, n_features)，表示输入数据，其中 n_samples 是样本数量，n_features 是特征数量。
                    - y：可选的整数数组，维度为(n_samples)，表示样本的类别标签。如果提供了 y，则会在预测时使用。
                    出参：
                    - y_pred：整数数组，维度为(n_samples)，表示预测的样本类别标签。预测的类别标签是根据聚类算法确定的，
                                      每个样本会被分配到距离最近的聚类中心所代表的类别中。
                                      数组中的元素表示每个数据点所属的簇的索引（从 0 开始, 3个簇就是 0 1 2）。
                    需要注意的是，`fit_predict()`方法是一个无参数方法，它会自动使用模型在训练数据上进行拟合，并对测试数据进行预测。
                    如果你想在预测之前对模型进行拟合，可以使用`fit()`方法；如果你想在预测之后获取聚类中心，可以使用`predict()`方法。
          因此，fit_predict()方法的作用是在数据上执行 KMeans 算法的拟合步骤，并使用拟合好的模型对新数据进行预测，返回聚类结果。
"""

clf = KMeans(n_clusters=3)
# 输出完整Kmeans函数，包括很多省略参数
print(clf)
'''
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=3, n_init=10, n_jobs=None, precompute_distances='auto',
       random_state=None, tol=0.0001, verbose=0)
'''


y_pred = clf.fit_predict(X)
# 输出聚类预测结果
print("y_pred = ", y_pred)
'''
数组中的元素表示每个数据点所属的簇的索引（从 0 开始, 3个簇就是 0 1 2）， 20个数据点 
y_pred =  [1 2 1 1 1 1 1 1 0 0 0 1 1 1 0 1 1 1 1 0 0]
'''



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
# plt.scatter(x, y, c=y_pred, marker='x')
scatter = plt.scatter(x, y, c=y_pred, marker='x')

# 绘制标题
plt.title("Kmeans-Basketball Data")
# 绘制x轴和y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
# 设置右上角图例
# plt.legend(["A", "B", "C"])
'''
将散点图的第一个图例元素添加到图例中，并为其指定标签为"A"、"B"和"C"。
scatter.legend_elements()返回一个包含散点图所有图例元素的列表，索引从 0 开始, 因此，handles=scatter.legend_elements()[0]表示将散点图的第一个图例元素添加到图例中。
'''
plt.legend(handles=scatter.legend_elements()[0], labels=["A", "B", "C"])  # matplotlib3.1.2可用
# 显示图形
plt.show()
