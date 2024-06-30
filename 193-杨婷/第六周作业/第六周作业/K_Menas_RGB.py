import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像
img = cv2.imread('lenna.png')
rows, cols = img.shape[0], img.shape[1]
# print(img)
# print(img.shape)

# 转换图像矩阵格式
data = img.reshape((-1, 3))
data = np.float32(data)
# print("--------------------------")
# print(data)

# 设置停止条件
criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类 聚集成2类
compactness2, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)

# K-Means聚类 聚集成4类
compactness4, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)

# K-Means聚类 聚集成8类
compactness8, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)

# K-Means聚类 聚集成16类
compactness16, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)

# K-Means聚类 聚集成64类
compactness64, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

# 图像转换回uint8二维类型
centers2 = np.uint8(centers2)  # 聚类中心，这里是2个
res2 = centers2[labels2.flatten()]  # 先把labels转换为一维数组，然后用两个聚类中心的像素值组成新的一维数组
dst2 = res2.reshape(img.shape)
# print(dst2)

centers4 = np.uint8(centers4)
res4 = centers4[labels4.flatten()]
dst4 = res4.reshape(img.shape)

centers8 = np.uint8(centers8)
res8 = centers8[labels8.flatten()]
dst8 = res8.reshape(img.shape)

centers16 = np.uint8(centers16)
res16 = centers16[labels16.flatten()]
dst16 = res16.reshape(img.shape)

centers64 = np.uint8(centers64)
res64 = centers64[labels64.flatten()]
dst64 = res64.reshape(img.shape)

# 图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = ['原始图像', '聚类图像 K=2', '聚类图像 K=4',
          '聚类图像 K=8', '聚类图像 K=16',  '聚类图像 K=64']
images = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
