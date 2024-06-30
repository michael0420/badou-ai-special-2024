import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像灰度颜色
img = cv2.imread('lenna.png', 0)  # 确保图像文件名和路径正确
if img is None:
    print("Error: 图像无法读取。")
    exit()
print(img.shape)

# 图像二维像素转换为一维
data = img.reshape((-1, 1))  # 使用-1自动计算行数
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# 设置初始中心选择为随机
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类
compactness, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, flags)

# 生成最终图像
dst = centers[labels.flatten()].reshape(img.shape)
dst = np.uint8(dst)

plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'), plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()