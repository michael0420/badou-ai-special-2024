import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import matplotlib

matplotlib.use('TkAgg')

img = cv.imread('../lenna.png')

data = img.reshape((-1, 3))
data = np.float32(data)
criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 10, 1.0)
flags = cv.KMEANS_RANDOM_CENTERS

compactness2, labels2, centers2 = cv.kmeans(data, 2, None, criteria, 10, flags)
compactness4, labels4, centers4 = cv.kmeans(data, 4, None, criteria, 10, flags)
compactness8, labels8, centers8 = cv.kmeans(data, 8, None, criteria, 10, flags)
compactness16, labels16, centers16 = cv.kmeans(data, 16, None, criteria, 10, flags)
compactness64, labels64, centers64 = cv.kmeans(data, 64, None, criteria, 10, flags)

# 图像转换回uint8二维类型
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
print(centers2)
print(labels2.flatten())
dst2 = res.reshape(img.shape)

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

# 图像转换为RGB显示
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
dst2 = cv.cvtColor(dst2, cv.COLOR_BGR2RGB)
dst4 = cv.cvtColor(dst4, cv.COLOR_BGR2RGB)
dst8 = cv.cvtColor(dst8, cv.COLOR_BGR2RGB)
dst16 = cv.cvtColor(dst16, cv.COLOR_BGR2RGB)
dst64 = cv.cvtColor(dst64, cv.COLOR_BGR2RGB)

plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16', u'聚类图像 K=64']
images = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.axis("off")
plt.show()
