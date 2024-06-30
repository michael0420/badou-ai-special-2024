import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

X = np.array([[1,2],[3,2],[4,4],[1,2],[1,3]])
'''
single:最小距离法，选择两个簇之间最小的距离作为两个簇直接的距离
complete: 最大距离法，最远临近法，选择两个簇之间最大的距离作为两个簇直接的距离
average：平均距离法，选择两个簇之间所有点对的平均距离作为两个簇之间的距离
weighted：加权平均距离法
centroid：质心法，两个簇的质心之间的距离来代表簇距离。
median: 中位数法
ward：最小方差法，选择是的簇内的总平方误差

'''
methods = ['single','complete','average','weighted','centroid','ward', 'median']
#绘制图像
fig, axes = plt.subplots(2,4,figsize=(10, 5))
axes = axes.flatten()

for i,method in enumerate(methods):
    z = linkage(X,method)
    max_d = 2
    clusters = fcluster(z, max_d, criterion='distance')
    print(f'Clusters for method {method}:{clusters}')
    #绘制树状图
    dendrogram(z,ax=axes[i])
    axes[i].set_title(f'method:{method}')
    if method == 'ward':
        print(z)
for j in range(len(methods),len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()


