'''
【第7周作业】
1实现层次聚类
'''


# 导入相应的包
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。
'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''
# 创建数据集
sample = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [7, 1],
    [5, 8],

]
# 获取聚类步骤数据
stepdata=linkage(sample,"average")
# print(stepdata)
# 获取聚类结果
result=fcluster(stepdata,10,"distance")
# print(result)
# 画图显示
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(stepdata)
print(dn)
plt.show()