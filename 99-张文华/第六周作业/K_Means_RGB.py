'''实现对图像的KMeans'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')

# 将数组从二维转为一维
data = img.reshape((-1,3))
data = np.float32(data)
print(data)

# 设置停止条件
stop = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 10, 1)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K—means
compactness, labels, centers = cv2.kmeans(data, 10, None, stop, 10, flags)

# 查看返回后的图像
centers = np.uint8(centers)
res = centers[labels.flatten()]
dst = res.reshape((img.shape))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()