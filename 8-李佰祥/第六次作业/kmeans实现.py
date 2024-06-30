import numpy as np
import  matplotlib as plt
#
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../../lenna.png', 0)
print (img.shape)

h ,w = img.shape

#图像二维像素转换为一维
# (但图像要是以二维形式传入，每列就变成一个特征了，最后就只有512个分类结果了，转成一维传入才是给每个像素聚类)
data = img.reshape((h * w, 1))
data = np.float32(data)

#
criteria  =(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,10,1)


#
flags = cv2.KMEANS_RANDOM_CENTERS

#
compactness , labels, centers = cv2.kmeans(data,2,None,criteria,10,flags)
print(centers)
#
dst = labels.reshape(img.shape[0],img.shape[1] )
print(dst)

plt.rcParams['font.sans-serif']=['SimHei']
plt.subplot(1,2,1)
plt.imshow(dst,'gray')
plt.show()

























