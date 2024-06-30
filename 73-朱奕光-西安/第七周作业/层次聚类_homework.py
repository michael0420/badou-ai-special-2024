import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
import numpy as np

data = np.array([[1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9]])
la = linkage(data,method='ward')
labels = fcluster(la,3,'distance')
dn = dendrogram(la)
print(la)
plt.show()