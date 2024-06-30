import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]

# 1计算距离 得到Z矩阵

Z = linkage(X,'average')
'''
Z = linkage(X,'ward')  #离差平方和距离
Z = linkage(X,'centroid') #质心距离
Z = linkage(X,'single')  #最邻近
Z = linkage(X,'complete') # 最远
'''

f = fcluster(Z, 1.5,'distance')  # t=1.5：是阈值，表示从哪个值划分（纵坐标上）。 distance：按照距离标准进行划分

print(Z)
print(f)  # 每个元素属于哪种类别

plt.figure()        # 参数： figsize:图像长宽
dendrogram(Z)

plt.show()