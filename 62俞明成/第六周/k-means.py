import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import matplotlib

matplotlib.use('TkAgg')

img = cv.imread("../lenna.png", 0)
row, col = img.shape

data = img.reshape((row * col, 1))
data = np.float32(data)

criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 10, 1.0)
flags = cv.KMEANS_RANDOM_CENTERS

retval, bestLabels, centers = cv.kmeans(data, 4, None, criteria, 10, flags)
dst_img = bestLabels.reshape(row, col)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

titles = ['原始图像', '聚类图像']
images = [img, dst_img]
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.axis("off")
plt.show()
