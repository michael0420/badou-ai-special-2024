from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。
method = ‘single’
d(u,v) = min(dist(u[i],u[j]))
对于u中所有点i和v中所有点j。这被称为最近邻点算法。
method = 'complete’
d(u,v) = max(dist(u[i],u[j]))
对于u中所有点i和v中所有点j。这被称为最近邻点算法。
method = 'average’
这被称为UPGMA算法（非加权组平均）法。
method = 'weighted’
d(u,v) = (dist(s,v) + dist(t,v))/2
u是由s和t形成的，而v是森林中剩余的聚类簇，这被称为WPGMA（加权分组平均）法。
method = ‘ward’ （沃德方差最小化算法）
'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
Z代表了利用“关联函数”关联好的数据。
比如上面的调用实例就是使用欧式距离来生成距离矩阵，并对矩阵的距离取平均
这里可以使用不同的距离公式
t这个参数是用来区分不同聚类的阈值，在不同的criterion条件下所设置的参数是不同的。
比如当criterion为’inconsistent’时，t值应该在0-1之间波动，t越接近1代表两个数据之间的相关性越大，t越趋于0表明两个数据的相关性越小。这种相关性可以用来比较两个向量之间的相关性，可用于高维空间的聚类
depth 代表了进行不一致性(‘inconsistent’)计算的时候的最大深度，对于其他的参数是没有意义的，默认为2
criterion这个参数代表了判定条件，这里详细解释下各个参数的含义：
（1）当criterion为’inconsistent’时，t值应该在0-1之间波动，t越接近1代表两个数据之间的相关性越大，t越趋于0表明两个数据的相关性越小。这种相关性可以用来比较两个向量之间的相关性，可用于高维空间的聚类
（2）当criterion为’distance’时，t值代表了绝对的差值，如果小于这个差值，两个数据将会被合并，当大于这个差值，两个数据将会被分开。
（3）当criterion为’maxclust’时,t代表了最大的聚类的个数，设置4则最大聚类数量为4类，当聚类满足4类的时候，迭代停止
（4）当criterion为’monocrit’时，t的选择不是固定的，而是根据一个函数monocrit[j]来确定。
'''

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
z = linkage(X,'ward')   #进行层次聚类
f = fcluster(z,t=4,criterion='distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(z)	#层级聚类结果以树状图表示出来并保存

print(z)
plt.show()

