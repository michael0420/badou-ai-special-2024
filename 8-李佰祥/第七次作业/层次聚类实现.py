import cv2
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

#优点是不用设置k值且可以发现类别之间的层次关系
#如果数据分布服从高斯分布，那么优先选择kmeans，因为从高斯分布的图来看有明显的中心点和簇结构
#与kmeans的工作原理匹配
#缺点：占用计算资源大
#计算每个样本之间的距离，合并距离最近的点形成一个组，每合并一次，再计算这个组和其他样本的距离，最后形成一个大的类
#计算距离： 最短距离法
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]

Z = linkage(X,'ward')
print(Z)
f = fcluster(Z,4,'distance') #进行四次聚类结果
print(f)
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()


























