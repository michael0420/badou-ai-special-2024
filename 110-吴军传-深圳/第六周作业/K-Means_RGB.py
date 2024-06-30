import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像,不是灰度图像
img = cv2.imread('lenna.png')
print(img.shape)

# #获取图像高度 宽度
# rows,cols = img.shape[:]

# 图像二维转为一维
data = img.reshape((-1, 3))
data = np.float32(data)

# 停止条件
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类  聚集成2类
compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
# K-Means聚类  聚集成4类
compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
# K-Means聚类  聚集成8类
compactness, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
# K-Means聚类  聚集成16类
compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
# K-Means聚类  聚集成64类
compactness, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

# #生成最终图像
# dst = labels.reshape((img.shape[0],img.shape[1]))

# 图像转换成uint8二维类型
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape(img.shape)

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape(img.shape)

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape(img.shape)

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape(img.shape)

# 图像转换成RGB显示
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img4 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img8 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img16 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img64 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4', u'聚类图像 K=8', u'聚类图像 K=16', u'聚类图像 K=32']
images = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
