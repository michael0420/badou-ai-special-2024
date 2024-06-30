from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
import matplotlib.pyplot as plt

'''
 linkage的第一个参数是距离矩阵，可以是1维压缩向量，也可以是2维观测向量
 第二个是计算类间距离的方法 
 
 fcluster第一个参数是linkage得到的矩阵，记录了层次聚类的层次信息 画图用的
    第二个参数一个聚类的阈值
 '''

X=[[1,2],[3,2],[4,4],[1,2],[1,3]]
Z = linkage(X,'ward')
f =fcluster(Z,4,'distance')
fig =plt.figure(figsize=(5,3))
dn = dendrogram(Z)
print(Z)
plt.show()
