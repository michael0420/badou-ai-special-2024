import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster

data = np.random.randint(1,10, size=(10, 2))
print(data)
Z = linkage(data, 'ward')
f = fcluster(Z, 4, 'distance')
plt.figure(figsize=(5,4))
dendrogram(Z)
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel('样本索引')
plt.ylabel('样本距离')
plt.title('层次聚类树状图')
plt.show()

