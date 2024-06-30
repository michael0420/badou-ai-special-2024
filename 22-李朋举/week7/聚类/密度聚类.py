import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import datasets 
from  sklearn.cluster import DBSCAN
 
iris = datasets.load_iris() 
X = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
'''
    从一个名为 `iris` 的数据集中提取特征数据的。`iris` 通常指的是鸢尾花数据集（Iris dataset），它是一个经典的小型数据集，常用于分类任务，
    特别是用于机器学习和统计教学。这个数据集包含了三种不同的鸢尾花（Setosa、Versicolour 和 Virginica）的四个特征：
                         花萼长度（sepal length）、花萼宽度（sepal width）、花瓣长度（petal length）和花瓣宽度（petal width）。
    1. `iris.data`：这部分表示 `iris` 数据集中的特征数据。它通常是一个二维数组（或类似结构，如 pandas DataFrame 的列），其中每一行代表一个样本，每一列代表一个特征。
    2. `[:, :4]`：这是一个 NumPy 风格的切片操作，用于从二维数组中选择特定的行和列。
        * `:`：第一个冒号表示选择所有行。
        * `:4`：第二个冒号后面的 `4` 表示选择从第一列到第四列（不包括第四列之后的列）。
    因此，`X = iris.data[:, :4]` 这行代码的作用是从 `iris` 数据集中提取所有样本的前四个特征（即花萼长度、花萼宽度、花瓣长度和花瓣宽度），并将这些特征赋值给变量 `X`。
'''
print(X.shape)  # (150, 4)

# 绘制数据分布图
'''
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')  
plt.xlabel('sepal length')  
plt.ylabel('sepal width')  
plt.legend(loc=2)  
plt.show()  
'''

dbscan = DBSCAN(eps=0.4, min_samples=9)  # 创建了一个 `DBSCAN` 聚类器对象，并设置了两个关键的参数
dbscan.fit(X)  # 聚类器对象对输入数据 `X` 进行聚类的命令
label_pred = dbscan.labels_  # 获取每个样本的聚类标签
'''
`DBSCAN` 是一种基于密度的聚类算法，全称为“Density-Based Spatial Clustering of Applications with Noise”。
在给定一个数据集时，`DBSCAN` 能够发现任意形状的聚类，并且能够识别出噪声点（即不属于任何聚类的点）。
    dbscan = DBSCAN(eps=0.4, min_samples=9) 这行代码创建了一个 `DBSCAN` 聚类器对象，并设置了两个关键的参数：
    1. `eps=0.4`: 这是一个距离阈值。在 `DBSCAN` 算法中，两个样本点被认为是邻居，如果它们之间的距离小于或等于 `eps`。
    2. `min_samples=9`: 这是一个密度阈值。对于数据集中的每个点，它的 `eps`-邻域中至少需要包含 `min_samples` 个点（包括该点自身），这个点才能被视为一个核心点。
    核心点周围的其他点（如果它们也在这个 `eps`-邻域内）会被归并到与核心点相同的聚类中。
        这两个参数的选择对于 `DBSCAN` 的聚类结果至关重要。`eps` 的值太小可能会导致大多数点都被视为噪声，而 `eps` 的值太大则可能导致所有的点都被归并到同一个聚类中。
        同样，`min_samples` 的值太小可能会导致算法将非常小的、紧密的聚类识别为噪声，而值太大则可能使得算法无法识别出任何聚类。
        在实际应用中，这些参数通常需要根据数据集的具体情况进行调整，以便得到最佳的聚类结果。这可能需要一些试验和错误，以及对数据集的深入了解。
        
    `dbscan.fit(X)` 是使用 `DBSCAN` 聚类器对象对输入数据 `X` 进行聚类的命令。
    `fit` 方法是 `sklearn`（一个流行的 Python 机器学习库）中许多聚类算法（如 `KMeans`, `AgglomerativeClustering`, `DBSCAN` 等）共有的方法。
          这个方法用于计算聚类，并将结果存储在模型对象中，供后续分析和使用。
       具体来说，对于 `DBSCAN` 算法，`fit` 方法会遍历数据集中的每个点，并根据前面设置的 `eps` 和 `min_samples` 参数来确定哪些点属于核心点，哪些点属于边界点，
       以及哪些点被视为噪声（即不属于任何聚类）。然后，它会基于这些点的密度关系来构建聚类。
                    `X` 通常是一个二维数组或类似结构（如 pandas DataFrame），其中每一行代表一个样本，每一列代表一个特征。
                    例如，处理鸢尾花数据集（Iris dataset），那么 `X` 包含 150 行（样本数）和 4 列（花萼长度、花萼宽度、花瓣长度和花瓣宽度）。
                    执行 `dbscan.fit(X)` 后，您可以使用 `dbscan.labels_` 属性来获取每个样本的聚类标签。噪声点通常会被标记为 `-1`。
'''
 
# 绘制结果
x0 = X[label_pred == 0]  # 根据 DBSCAN 聚类结果筛选出属于聚类标签为 0 的所有样本点。
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
'''
X 是原始数据集，一个二维数组或类似结构，其中每一行代表一个样本，每一列代表一个特征。label_pred 是一个一维数组，包含了 DBSCAN 算法为每个样本点分配的聚类标签。
表达式 label_pred == 0 会生成一个布尔数组，其中与 label_pred 中值为 0 的位置相对应的元素为 True，其他位置为 False。然后，这个布尔数组被用作索引来从 X 中选择行。
因此，x0 将是一个二维数组，只包含那些被 DBSCAN 算法分配到聚类标签为 0 的样本点。如果数据集中有多个聚类，并且其中一个是标签为 0 的聚类，那么 x0 将包含该聚类的所有样本点。
这样的筛选操作在聚类分析后可以进一步探索或分析特定聚类的特性。
'''
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')   # 绘制一个散点图，这个散点图展示了 x0 中所有样本点的分布情况
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')  
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
'''
使用 matplotlib 的 scatter 函数来绘制一个散点图，这个散点图展示了 x0 中所有样本点的分布情况。
这里，x0 是一个二维数组，包含了所有聚类标签为 0 的样本点。x0[:, 0] 和 x0[:, 1] 分别表示 x0 中所有样本点的第一个和第二个特征的值。
    c="red"：设置散点的颜色为红色。
    marker='o'：设置散点的形状为圆圈。
    label='label0'：为这些散点设置一个标签，这样在添加图例（legend）时可以显示这个标签。
可视化聚类标签为 0 的样本点。通过将颜色设置为红色并使用圆圈作为标记，您可以更清楚地看到这些点的分布。
'''
plt.xlabel('sepal length')  # x轴
plt.ylabel('sepal width')  # y轴
plt.title('sepal')  # 标题
plt.legend(loc=2)   # 显示图例
plt.show()  # 显示图形

