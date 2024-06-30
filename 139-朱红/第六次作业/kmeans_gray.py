import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
# 二维图像转一维
img_one = img.reshape((-1,1))
img_one = np.float32(img_one)

# 停止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

# Kmeans聚类
compactness, labels, centers = cv2.kmeans(img_one, 4, None, criteria, 10, flags)

# 转换为uint8类型
centers = np.uint8(centers)
res = centers[labels.flatten()]
dst = res.reshape(img.shape)

# 正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

# 显示图像
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("原始图像")
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(dst, cmap='gray')
plt.title("聚类后图像")
plt.xticks([]), plt.yticks([])

plt.show()