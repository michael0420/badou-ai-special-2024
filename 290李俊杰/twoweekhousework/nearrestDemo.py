# 【第二周作业】
#
# 2.实现最临近插值

import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取一个512*512的图片使用临近插值放大2倍变成1023*1023的图片
# 在测试当中发现如果使用1024*1024或者更大需要去掉0.5，不然会超出索引值
img=cv2.imread("lenna.png")
h,w,channels=img.shape
# 创建一个尺寸为1023*1023，像素值为0的图片
# emptyImage=np.zeros((800,800,channels),np.uint8)
emptyimg=np.zeros((1023,1023,channels),np.uint8)
# 求放大后的比例
sh=1023/h
sw=1023/w
for i in range(1023):
    for j in range(1023):
        # x,y是求新图片的每个像素值应该取原图片上哪个位置的像素值
        # 使用临近插值法求出新图片每个坐标位置接近原图的哪个坐标
        #取出像素值插入新图
        # x,y为索引值，必须为整数
        # int为向下取整可能导致
        # [4.6,4,8]接近[5,5]，应取[5,5]的像素值
        # 但是取的是[4,4]坐标上的像素值，所以要后面加上0.5变成类似四舍五入
        x=int(i/sh+0.5)
        y=int(j/sw+0.5)
        emptyimg[i,j]=img[x,y]
print(emptyimg)
cv2.imshow("image show bigimg",emptyimg)
cv2.waitKey(0)
