import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

X = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9, 35],
              [42, 45, 11],
              [9, 48, 5],
              [11, 21, 14],
              [8, 5, 15],
              [11, 12, 21],
              [21, 20, 25]])
pca = PCA(n_components=2)
# pca.fit(X)                  #训练
X_pca =pca.fit_transform(X)
print(X_pca)
# y =[1,2,3,4,5,6,7,8,9,10]
# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA of Iris Dataset')
# plt.colorbar(label='Species')
# plt.show()