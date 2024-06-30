from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。如 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 或 'ward'。


'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是由linkage函数计算得到的层次聚类链接矩阵，它包含了所有数据点之间的合并信息。
2.t  是一个切割阈值，用于决定何时将树剪枝，生成聚类。
3.criterion：用于确定如何切割树的准则。常见的选项有：
    'distance'：基于最大距离，当两个子簇之间的距离大于t时，它们被分配到不同的集群。
    'maxclust'：指定要生成的聚类的最大数量。
返回值：fcluster函数返回一个整数数组，其中每个元素表示相应数据点所属的聚类ID。
        这个数组可以用于进一步的数据分析，比如与原始数据合并，以便查看每个数据点属于哪个聚类。
'''

# 定义输入数据X。每个元素表代表一个数据点的坐标。
X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]

# 使用ward方法进行层次聚类，得到层次聚类链接矩阵
Z = linkage(X, 'ward')
# 打印Z，展示所有数据点之间的合并信息
print('层次聚类链接矩阵:')
print(Z)

f = fcluster(Z, 2, 'maxclust')
print('聚类结果:')
print(f)

# 创建绘图对象并设置尺寸
plt.figure(figsize=(5, 4))

# 绘制dendrogram
dendrogram(Z)

# 设置字体为SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置x轴和y轴标签
plt.xlabel('数据序号')
plt.ylabel('第Y步')
plt.show()
