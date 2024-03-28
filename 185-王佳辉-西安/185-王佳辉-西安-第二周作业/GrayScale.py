# -*- coding: utf-8 -*-
from skimage.color import rgb2gray
import numpy as np
import  matplotlib.pyplot as plt
import cv2
from PIL import Image  #可以在一个画布中呈现多张图像
from skimage.color import rgb2gray


####灰度化
img = cv2.imread("img/hutao.jpg")#读取图片
h,w = img.shape[:2]#在这个img数组中取前两位的值为0和1的值，获取图片的长宽，长：high，宽：wide
img_gray = np.zeros([h,w],img.dtype)#设置一张同等长宽的单通道空图片：创建一张与当前图片一致的单通道图片
#循环遍历获取high和wide对应的坐标值
for i in range(h):
    for j in range(w):
        m = img[i,j]        #获取当前high和wide的rgb值
        img_gray[i,j]= int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3) #通过计算公式将bgr转化为gray坐标并赋值给新图像，m[0]：B ， m[1]：g  m[2]：r


print(m)#输出原图片的rgb矩阵值
print(img_gray)#输出转化为二值化图片的rgb矩阵值
print("img show gray: %s"%img_gray)
cv2.imshow("gray",img_gray)#展示灰度化图片
cv2.imwrite("img/gray_hutao.jpg",img_gray)#将生成的图片保存下来


plt.subplot(221)#在2*2画布上在第一个位置上显示img
img =plt.imread("img/hutao.jpg")
plt.imshow(img)
print("------image lenna-------")
print(img)

# 灰度化：调用接口，灰度化的两种接口
img_gray = rgb2gray(img)#skimage.color，rgb2gray
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #cv2.cvtColor

plt.subplot(222)#在2*img_gray
plt.imshow(img_gray,cmap='gray')#表明展示的为灰度图
print("---image gray----")
print(img_gray)

# 二值化
rows,cols = img_gray.shape#灰度图的长宽
for i in range(rows):
    for j in range(cols):
        if(img_gray[i,j] <= 0.5):
            img_gray[i,j] = 0
        else:
            img_gray[i,j] = 1
img_binary = np.where(img_gray > 0.5,1,0)
cv2.imwrite("img/binary_hutao.jpg",img_gray)#将生成的图片保存下来
print("----image binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)#在2*2画布上在第三个位置上显示img_binary
plt.imshow(img_binary, cmap='gray')
plt.show()#加上这句才会把图片展示出来
