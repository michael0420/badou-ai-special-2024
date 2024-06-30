#利用open cv中  实现聚类

import  cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('lenna.jpg',1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('i',img)
# cv2.waitKey()
print(img.shape)
rows,cols = img.shape[:]

#转换为一维图像
data = img.reshape((rows*cols,1))
data = np.float32(data)

#设置停止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS
#主函数
compacteness,labels,centers = cv2.kmeans(data,5,
None,criteria,10,flags)
dst = labels.reshape((img.shape[0],img.shape[1]))

plt.rcParams['font.sans-serif']=['SimHei']
titles = [u'原始图像', u'聚类图像']
images= [img,dst]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray'),
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
