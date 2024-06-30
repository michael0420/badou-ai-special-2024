import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('lenna.png', 0)
print (img.shape)
rows, cols = img.shape[:]
#图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags=cv2.KMEANS_RANDOM_CENTERS
compactnes,labels,centers=cv2.kmeans(data,4,None,criteria,10,flags)
dst=labels.reshape((img.shape[0],img.shape[1]))
#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()
