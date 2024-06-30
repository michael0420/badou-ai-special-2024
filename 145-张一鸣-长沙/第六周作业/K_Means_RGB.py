# coding = utf-8

'''
        实现彩图K-Means聚类
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')
print(img.shape)

# 将图像数据转换为一维
one = img.reshape(-1, 3)        # img.reshape((-1,3))将图像数组转换为一维数组
one = np.float32(one)

# 设置终止模式
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 0)

# labels数组，每个元素是数据点所属集群的标签（从 0 到 K-1）
# centers数组，包含每个集群的最终质心位置

# K-Means聚类 聚集成2类
_, labels2, centers2 = cv2.kmeans(one, 2, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)

# K-Means聚类 聚集成4类
_, labels4, centers4 = cv2.kmeans(one, 4, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)

# K-Means聚类 聚集成8类
_, labels8, centers8 = cv2.kmeans(one, 8, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)

# K-Means聚类 聚集成16类
_, labels16, centers16 = cv2.kmeans(one, 16, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)

# K-Means聚类 聚集成32类
_, labels32, centers32 = cv2.kmeans(one, 32, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)

# K-Means聚类 聚集成64类
_, labels64, centers64 = cv2.kmeans(one, 64, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)

# K-Means聚类 聚集成128类
_, labels128, centers128 = cv2.kmeans(one, 128, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)

# 图像转为u-int8类型
# labels.flatten(): 这是一个NumPy数组的方法，用于将多维数组转换为一维数组
print(type(labels2), labels2)
centers2 = np.uint8(centers2)
# 使用labels.flatten()作为索引来从centers2中选择元素，
# 对于labels.flatten()中的每个标签（即每个像素的聚类索引），都会从centers中选择对应的聚类中心
# 结果res是一个一维数组，其长度与labels.flatten()相同，并且包含与原始图像中每个像素对应的聚类中心的坐标或值
res = centers2[labels2.flatten()]
print(type(res), res)
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape((img.shape))

centers32 = np.uint8(centers32)
res = centers32[labels32.flatten()]
dst32 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape((img.shape))

centers128 = np.uint8(centers128)
res = centers128[labels128.flatten()]
dst128 = res.reshape((img.shape))

# 图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst32 = cv2.cvtColor(dst32, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)
dst128 = cv2.cvtColor(dst128, cv2.COLOR_BGR2RGB)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4', u'聚类图像 K=8',
          u'聚类图像 K=16', u'聚类图像 K=32', u'聚类图像 K=64', u'聚类图像 K=128']
images = [img, dst2, dst4, dst8, dst16, dst32, dst64, dst128]
for i in range(8):
   plt.subplot(2, 4, i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]), plt.yticks([])
plt.show()
