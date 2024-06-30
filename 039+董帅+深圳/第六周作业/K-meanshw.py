import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像的灰度颜色
img = cv2.imread('lenna.png')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(img_gray.shape)

rows, cols = img_gray.shape[:2]
data = img_gray.reshape((rows*cols,1))
data = np.float32(data)

#停止条件
criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
flags=cv2.KMEANS_RANDOM_CENTERS

compactness, labels, centers=cv2.kmeans(data,4,None, criteria,10, flags)
dst=labels.reshape((img.shape[0], img.shape[1]))

plt.rcParams['font.sans-serif'] = ['SimHei']

title = [u'original image', u'k-means image']

images = [img_gray,dst]
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(images[i], 'gray'),
    plt.title(title[i])
    plt.xticks([]),plt.yticks([])
plt.show()



