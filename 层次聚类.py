from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
from matplotlib import pyplot as plt

"""
这段代码从scipy.cluster.hierarchy模块导入了三个函数:dendrogram()用于绘制树状图，linkage()用于计算层次聚类的层次信息
fcluster()用于将层次聚类转换为平面聚类。同时，从matplotlib的pyplot模块导入plt,用于绘图。
"""

"""
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。方法为单链法
3.metric="euclidean":指定了距离度量方式为欧几里得距离。
"""

'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值，用于形成平面聚类。
3.其他参数如criterion,depth等提供了更复杂的聚类选项，但在这个例子中未使用。
'''

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
Z = linkage(X, 'ward')
f = fcluster(Z,4,'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()