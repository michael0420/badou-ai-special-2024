import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像灰度颜色
img = cv2.imread("../data/lenna.png",0)
print (img.shape)

#获取图像高度、宽度
rows, cols = img.shape[:]

#图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

#停止条件 (type,max_iter,epsilon)


criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 100, None, criteria, 10, flags)
print(labels)
#生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))
print(dst)
print(img)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['Arial Unicode MS']

#显示图像
titles = [u'原始图像', u'聚类图像']
img2=cv2.equalizeHist(cv2.convertScaleAbs(dst))
print(img2)
images = [img, img2]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()