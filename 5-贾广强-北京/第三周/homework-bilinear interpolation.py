#!/usr/bin/env python
# coding: utf-8

# In[22]:


import cv2
#获取原始图像
o_image = cv2.imread('lenna.jpg')
# 定义函数及参数
def resized_image(image,dst_w,dst_h):
    #使用cv2实现双线性插值
    new_image = cv2.resize(image,(dst_w,dst_w),interpolation=cv2.INTER_LINEAR)
    return new_image
#调用函数并传入新的尺寸
r_image = resized_image(o_image,600,600)
#判断图像前后关系
if r_image.shape == o_image.shape:
    print('图像尺寸与原图一致',o_image.shape)
else:
    print('图像已被修改',r_image.shape)
#输入图像
cv2.imshow('original_image',o_image)
cv2.imshow('resized_image',r_image)
cv2.waitKey(0)


# In[27]:


import cv2
o_image = cv2.imread('lenna.jpg')
dst_w = 800
dst_h = 900

resized_image = cv2.resize(o_image,(dst_w,dst_h),interpolation=cv2.INTER_LINEAR)
print(resized_image.shape)
cv2.imshow('1',resized_image)
cv2.waitKey()

