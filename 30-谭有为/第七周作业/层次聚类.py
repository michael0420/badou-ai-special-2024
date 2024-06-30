import matplotlib.pyplot as plt
#scipy.cluster.hierarchy模块提供了一系列的函数和类来进行聚类分析，这包括使用各种方法构建聚类，以及展示聚类的层次结构。
#dendrogram函数用于绘制系统聚类的树状图。
#linkage函数用于计算聚类的系统聚类方法。
#fcluster函数用于将点分配给系统聚类。
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster

X=[[1,2],[2,3],[5,1],[4,4],[7,1],[9,9],[3,2]]
#linkage函数，调用格式 Z=linkage(Y,‘method’)，method：可取值有：‘single’：最短距离法（默认），
#‘complete’：最长距离法；‘average’：未加权平均距离法；‘weighted’： 加权平均法；‘centroid’：质心距离法；
#‘median’：加权质心距离法；‘ward’：内平方距离法（最小方差算法）
Z=linkage(X,'complete')  #使用linkage函数计算链接矩阵
print(Z)
f=fcluster(Z,4,'distance')
#使用fcluster函数将数据分配到不同簇中  4表示阈值，即控制聚类的停止条件  distance表示计算簇之间距离的准则 可选参数有：
#’inconsistent’、‘distance’、‘maxclust’和’monocrit’
print(f)
#根据聚类结果绘制图
#pic=plt.figure(figsize=(7,5))
dendrogram(Z)  #生成聚类树
plt.show()


