# 导入层次聚类函数、树状图函数和划分聚类函数
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt # 导入绘图库

# linkage函数用于计算层次聚类

'''
参数：
y：可以是距离矩阵，也可以是观测数据数组。如果是观测数据数组，则函数内部会首先计算距离矩阵。
method：聚类合并的方法，'ward' 表示使用 Ward 方法，即最小化合并后的总方差。
metric：计算距离的度量，默认是欧氏距离。

输出：
返回一个数组，描述了层次聚类的层次结构，该数组有 (n-1) 行，每行代表一个合并操作，包括两个合并簇的索引和距离，以及合并后簇的观测数。
'''
X = [[1,3],[2,4],[2,3],[1,2],[4,4],[5,2],[3,3],[2,2],[1,5]]  # 示例数据点，5个观测值，每个观测值是一个二维坐标
Z = linkage(X, "ward")  # 执行层次聚类，使用Ward方法

# fcluster函数用于从层次聚类结果中形成平面聚类
'''
参数：
Z：linkage函数的输出。
t：聚类时的距离阈值，小于等于此值的观测值会被归为同一个聚类。
criterion：聚类的准则，默认为'inconsistent'，本例中使用'distance'，即基于距离的阈值。
'''
f = fcluster(Z, 4, 'distance')  # 根据距离阈值4来划分聚类

# 绘制树状图展示聚类过程
fig = plt.figure(figsize=(5, 3))  # 创建一个图形对象，指定大小为5x3
dn = dendrogram(Z)  # 绘制树状图，基于层次聚类结果Z
print(Z)  # 打印层次聚类的输出结果，显示每次合并的详细信息

plt.show()  # 显示图形