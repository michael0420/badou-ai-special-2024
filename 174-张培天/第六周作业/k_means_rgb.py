import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
data = img.reshape((-1,3))
data = np.float32(data)

# 停止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

compactness2, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
compactness4, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
compactness8, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
compactness16, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
compactness64, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

# 转为图像
centers2 = np.uint8(centers2)
ret = centers2[labels2.flatten()]
dst2 = ret.reshape((img.shape))

centers4 = np.uint8(centers4)
ret = centers4[labels4.flatten()]
dst4 = ret.reshape((img.shape))

centers8 = np.uint8(centers8)
ret = centers8[labels8.flatten()]
dst8 = ret.reshape((img.shape))

centers16 = np.uint8(centers16)
ret = centers16[labels16.flatten()]
dst16 = ret.reshape((img.shape))

centers64 = np.uint8(centers64)
ret = centers64[labels64.flatten()]
dst64 = ret.reshape((img.shape))

# 显示RGB模式
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
img8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
img16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
img64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

# 展示图像
plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16',  u'聚类图像 K=64']  
images = [img, img2, img4, img8, img16, img64]  
for i in range(6):  
   plt.subplot(2,3,i+1)
   plt.imshow(images[i], 'gray')
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()
