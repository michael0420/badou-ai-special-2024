#!/usr/bin/env python
# coding: utf-8

# In[6]:


#导入库
import cv2
import numpy as np

def zoom_image(img):
    height, width, channels = img.shape #获取图像信息
    new_image = np.zeros ((new_height, new_width, channels), dtype = np.uint8) #创建新图像信息
    zoom_height = new_height / height
    zoom_width = new_width / width
    #遍历新图像的像素
    for i in range(800):
        for j in range(800):
        #计算对应的像素
           x = int (i / zoom_height + 0.5)
           y = int (j / zoom_width + 0.5)
           new_image[i,j] = img[x,y]
            
    return new_image

#读取原始图像信息
img = cv2.imread("lenna.png")

#设定目标尺寸
new_height = 800
new_width = 800

#调取函数
new_image = zoom_image(img)

#同时显示原始图像和新图像
cv2.imshow("Zoomed Image", new_image)
cv2.imshow("Original Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




