import numpy as np
from sklearn import preprocessing
import scipy.cluster.hierarchy as sch
import matplotlib.pylab as plt
import matplotlib
matplotlib.rc('font', family='Microsoft YaHei')
A=np.array([[0, 1, 3, 1, 3, 4],
 [3, 3, 3, 1, 2, 1],
 [1, 0, 0, 0, 1, 1],
 [2, 1, 0, 2, 2, 1],
 [0, 0, 1, 0, 1, 0]])
B = preprocessing.minmax_scale(A, axis=0)

# 距离计算方法 metric: correlation:相关系数 cosine:夹角余弦 euclidean:欧氏距离
# 簇间距离判断 method: single:最短距离法 complete:最长距离法 average:平均距离法
Z=sch.linkage(B.T, metric='euclidean', method='single')  # 按B的行聚类
tree = sch.cut_tree(Z, height=0.5)
print(tree)
for i in tree:
    print(i)
    for j in i:
        print(j)