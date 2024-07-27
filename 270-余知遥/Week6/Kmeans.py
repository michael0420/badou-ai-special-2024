import cv2
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(42)
X = np.vstack((
    np.random.multivariate_normal([1, 1], [[0.1, 0], [0, 0.1]], 100),
    np.random.multivariate_normal([5, 5], [[0.1, 0], [0, 0.1]], 100),
    np.random.multivariate_normal([8, 1], [[0.1, 0], [0, 0.1]], 100)
)).astype(np.float32)

# 设置KMeans聚类参数
K = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
attempts = 10

# 运行KMeans算法
ret, labels, centers = cv2.kmeans(X, K, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

# 打印结果
print("Labels:", labels.ravel())
print("Centers:", centers)

# 可视化结果
labels = labels.flatten()
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=25)
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='x')
plt.title('K-Means Clustering')
plt.show()


# 读取图像
image = cv2.imread('lena_std.tif')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 将图像数据转换为二维数组
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# 设置KMeans聚类参数
K = 8  # 将颜色减少到8种
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
attempts = 10

ret, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

# 将中心点转换为8位无符号整型
centers = np.uint8(centers)
# 将每个像素标签替换为中心点的值
segmented_image = centers[labels.flatten()]
# 将数据重塑回原始图像的形状
segmented_image = segmented_image.reshape(image.shape)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('K-Means Color Compression 8')
plt.axis('off')
plt.show()
