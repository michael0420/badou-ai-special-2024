import numpy as np
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt

# 创建一些随机数据点

np.random.seed(5)
data = np.random.randn(10,2)

Z = linkage(data,method='ward')
print(Z)
plt.figure()
dendrogram(Z)
plt.show()
