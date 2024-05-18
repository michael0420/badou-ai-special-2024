# python 学习开始时间2021.09.12
import numpy as np
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image

img=cv2.imread("lenna.png")  #读取照片 照片存储路径层跟py文件是同一个路径层
h,w,channels =img.shape  #获取图片高和宽和渠道
img_up=np.zeros((800,800,channels),np.uint8) #创建一张800*800新照片
ph=800/h #高的缩放比例
pw=800/w #宽的缩放比例

for x in range(800):
    for y in range(800):
        i=int(x/ph+0.5)
        j=int(y/pw+0.5)# int(),转为整型，使用向下取整。
        img_up[x,y]=img[i,j]


print("------img-------")
print(img.shape)
print(img)
print("------img_up-------")
print(img_up.shape)
print(img_up)

cv2.imshow("nearest interp", img_up)
cv2.imshow("image", img)
cv2.waitKey(0)

