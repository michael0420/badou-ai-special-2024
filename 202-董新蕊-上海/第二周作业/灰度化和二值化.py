#!/usr/bin/env python
# coding: utf-8

# In[17]:


#灰度化
'''
方法一：利用opencv读取图像信息，使用加权平均方法计算灰度值
'''

#导入库
import cv2 
import numpy as np  

#读取图像像素信息及尺寸信息，创建与原始图像相同的新图像
img= cv2.imread("lenna.png")
height, width = img.shape[:2] 
new_image = np.zeros((height, width,3), dtype = np.uint8)

#遍历每个像素 
for i in range(height):
    for j in range(width):
        # 使用加权平均方法计算灰度值
        new_image[i,j] = int (0.11 * img[i,j][0] + 0.59 * img[i,j][1] + 0.3 * img[i,j][2])


cv2.imshow("Gray Image", new_image)
cv2.imshow("Original Image", img)


cv2.waitKey(0)
cv2.destroyAllWindows()

'''
方法二：利用skimage自带函数rgb2gray

import skimage.color 
from skimage.color import rgb2gray
image_gray = rgb2gray(img)

方法三：利用opencv自带函数
import cv2 
import numpy as np  
img= cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
cv2.imshow('gray image', img_gray)

cv2.waitKey(0)
'''

#二值化
'''
方法一：利用opencv自带函数
读取灰度图像信息，将图像信息二值化
'''
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary Image", img_binary)


cv2.waitKey(0)
cv2.destroyAllWindows()


'''
方法二：遍历图像像素值，
读取灰度图像信息，将图像信息二值化


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
height, width = img_gray.shape[:2] 
img_binary = np.zeros((height, width), dtype = np.uint8)
img_binary = img_gray.astype(np.float)/255.0

for i in range(height):
    for j in range(width):
        if img_binary[i,j] <= 0.5:
            img_binary[i,j] = 0
        else:
            img_binary[i,j] = 1

cv2.imshow("Binary Image", img_binary)
cv2.imshow("Gray Image", img_gray)
cv2.imshow("Original Image", img)

'''


# In[16]:





# In[ ]:




