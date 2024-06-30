from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

# 示例数据
x = [[1, 2], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]
z = linkage(x, 'single')
f = fcluster(z, 2, 'distance')
# 创建画布
plt.figure(figsize=(5, 3))
# 绘制图像
dendrogram(z)

print(z)
plt.show()
