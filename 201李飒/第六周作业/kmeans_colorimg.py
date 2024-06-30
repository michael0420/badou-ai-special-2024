import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
print(img.shape)

data = img.reshape((-1,3))
data = np.float32(data)
print(data)

# 停止条件
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

# 聚类
compactness,lables2,centers2 = cv2.kmeans(data,2,None,criteria,10,flags)
compactness,lables4,centers4 = cv2.kmeans(data,4,None,criteria,10,flags)
compactness,lables8,centers8 = cv2.kmeans(data,8,None,criteria,10,flags)
compactness,lables16,centers16 = cv2.kmeans(data,16,None,criteria,10,flags)
compactness,lables64,centers64 = cv2.kmeans(data,64,None,criteria,10,flags)

# 画图
centers2 = np.uint8(centers2)
res = centers2[lables2.flatten()]
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[lables4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[lables8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[lables16.flatten()]
dst16 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[lables64.flatten()]
dst64 = res.reshape((img.shape))

# 图像转化为RGB显示
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2,cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4,cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8,cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16,cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64,cv2.COLOR_BGR2RGB)

# 中文显示标题
plt.rcParams["font.sans-serif"]=["SimHei"]

# 显示图像
title = [u'原图',u"聚类图像，k=2",u"聚类图像，k=4",u"聚类图像，k=8",u"聚类图像，k=16",u"聚类图像，k=64",]
imgs = [img,dst2,dst4,dst8,dst16,dst64]
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(imgs[i],'gray')
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])
plt.show()