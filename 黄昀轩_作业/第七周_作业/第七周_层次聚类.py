"""
层次聚类
"""
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
import matplotlib.pyplot as plt
import numpy as np

X = np.random.normal(size=(10, 2))

lx = linkage(X,"ward")
y = fcluster(lx, 5,"distance") #t： 给定聚类得最大分层
plt.figure(figsize=(10,5))

dy = dendrogram((lx))
plt.show()
