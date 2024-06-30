import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 加载手写数字数据集
digits = load_digits()
X = digits.data
y = digits.target

# 绘制降维前的图像
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(10, 5))
ax = ax.flatten()
for i in range(10):
    img = X[y == i][0].reshape(8, 8)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# PCA 的实现过程
mean_vec = np.mean(X, axis=0)
cov_mat = np.cov(X.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
sorted_idx = np.argsort(eig_vals)[::-1]
sorted_eig_vals = eig_vals[sorted_idx]
sorted_eig_vecs = eig_vecs[:, sorted_idx]
k = 2
projection_mat = sorted_eig_vecs[:, :k]
X_pca = X.dot(projection_mat)

# 绘制降维后的图像
colors = ['red', 'blue', 'green', 'purple', 'yellow', 'orange', 'pink', 'cyan', 'gray', 'brown']
plt.figure(figsize=(10, 8))
for i in range(len(colors)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=colors[i], label=str(i), edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Digit')
plt.title('PCA on MNIST Dataset')
plt.show()