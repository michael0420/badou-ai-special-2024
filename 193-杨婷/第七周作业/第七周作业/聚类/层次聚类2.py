import numpy as np
from sklearn import preprocessing
import scipy.cluster.hierarchy as sch
import matplotlib.pylab as plt
import matplotlib
matplotlib.rc('font', family='Microsoft YaHei')
A = np.array([[0, 1, 3, 1, 3, 4],
  [3, 3, 3, 1, 2, 1],
 [1, 0, 0, 0, 1, 1],
 [2, 1, 0, 2, 2, 1],
 [0, 0, 1, 0, 1, 0]])
B = preprocessing.minmax_scale(A, axis=0)

# 距离计算方法 metric: correlation:相关系数 cosine:夹角余弦 euclidean:欧氏距离
# 簇间距离判断 method: single:最短距离法 complete:最长距离法 average:平均距离法
Z = sch.linkage(B.T, metric='euclidean', method='single')  # 按B的行聚类

# 给定距离阈值下的聚类结果
yuzhi = 0.5
label = []
for i in sch.cut_tree(Z, height=yuzhi):
    for j in i:
        label.append(j)
print('阈值='+str(yuzhi)+' 簇数='+str(len(list(set(label)))))
print('聚类簇 样本编号')
for i in list(set(label)):
    print(i, ' : ', end='\t')
    for j in range(len(label)):
        if i == label[j]:
            print(j, end='\t')
        else:
            pass
    print()
print('-'*15)

# 给定簇数下的聚类结果
n = 3
label = []
for i in sch.cut_tree(Z, n_clusters=n):
    for j in i:
        label.append(j)
print('簇数='+str(n))
print('聚类簇 样本编号')
for i in list(set(label)):
    print(i, ' : ', end='\t')
    for j in range(len(label)):
        if i == label[j]:
            print(j, end='\t')
        else:
            pass
    print()

# 画出聚类谱系图
sch.dendrogram(Z)
plt.show()