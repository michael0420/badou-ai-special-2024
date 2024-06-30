# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像
img = cv2.imread('img.png')
print(img.shape)

# 图像二维像素转换为一维
data = img.reshape((-1, 3))
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 设置标签
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16', u'聚类图像 K=64']

# 图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(2, 3, 1), plt.imshow(img, 'gray'),
plt.title(titles[0])
plt.xticks([]), plt.yticks([])

flags = cv2.KMEANS_RANDOM_CENTERS
for i, k in enumerate((2, 4, 8, 16, 64)):
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)

    # 图像转换回uint8二维类型
    centers2 = np.uint8(centers)
    res = centers2[labels.flatten()]
    dst = res.reshape((img.shape))
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, i + 2), plt.imshow(dst, 'gray'),
    plt.title(titles[i + 1])
    plt.xticks([]), plt.yticks([])

plt.show()
