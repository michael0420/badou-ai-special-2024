import cv2
import numpy as np
import matplotlib.pyplot as plt
# read the image
image = cv2.imread("me.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像重塑为像素和3个颜色值（RGB）的2D数组
print(image.shape) #(853, 1280, 3)
pixel_values = image.reshape((-1, 3))
# 转换为numpy的float32
pixel_values = np.float32(pixel_values)
print(pixel_values.shape) #(1091840, 3)

# 确定停止标准
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.1)

k = 5
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 转换回np.uint8
centers = np.uint8(centers)

# 展平标签阵列
labels = labels.flatten()

segmented_image = centers[labels.flatten()]

#重塑回原始图像尺寸
segmented_image = segmented_image.reshape(image.shape)
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(segmented_image)
plt.show()

# # 禁用2号群集（将像素变为黑色）
# masked_image = np.copy(segmented_image)
# # 转换为像素值向量的形状
# masked_image = masked_image.reshape((-1, 3))
# cluster1 = 1
# masked_image[labels == cluster1] = [0, 0, 0]
# # 转换回原始形状
# masked_image = masked_image.reshape(image.shape)
# plt.subplot(121)
# plt.imshow(image)
# plt.subplot(122)
# plt.imshow(masked_image)
# plt.show()
