import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png",0)
img = np.float32(img)
data = img.reshape(-1,1)

# 停止条件
critesia = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# Kmeans聚类成4类
compactiness, labels, centers = cv2.kmeans(data,4, None, critesia,10,flags)

dst = labels.reshape((img.shape[0], img.shape[1]))

# 显示图像
plt.rcParams['font.sans-serif'] = ["SimHei"]
titles = [u'原始图像',u'聚类图像']
images = [img, dst]
for i in range(len(images)):
    plt.subplot(1,2,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
