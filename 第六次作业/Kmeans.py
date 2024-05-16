#!/usr/bin/env python
# coding: utf-8

# In[32]:


# retval，bestlabels，criteria =cv2.kmeans(data,bestlabels,criteria,attemps,flags)
import cv2
import matplotlib.pyplot as plt
import numpy as np
# 读取图像、获取图像的宽度、高度
img =cv2.imread('lenna.png',0)
rows,cols =img.shape[:]
# 图像的二维数据一维化
data =img.reshape((rows*cols,1))
data =np.float32(data)
# 停止条件
criteria =(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,0.1)
# 初始中心选择
flags =cv2.KMEANS_RANDOM_CENTERS

# 聚类分析
retval,labels,criteria =cv2.kmeans(data,2,None,criteria,10,flags)

# 生成图像
dst =labels.reshape((img.shape[0],img.shape[1]))

# 用中文标签显示
plt.rcParams['font.sans-serif']=['SimHei']

# 显示图像
titles =[u'原始图像',u'聚类图象']
# images =[img,dst]
# for i in range(0,2):
#     plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
plt.subplot(121)
plt.title('原始图像')
plt.imshow(img,cmap ='gray')

plt.subplot(122)
plt.title('聚类图象')
plt.imshow(dst,cmap ='gray')
# plt.show()
    


# In[ ]:


from sklearn.cluster import KMeans

