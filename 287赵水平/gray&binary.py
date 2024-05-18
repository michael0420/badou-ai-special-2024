#2024.4.26 AI培训第二周作业灰度处理二值化处理
import numpy as np
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image

img=cv2.imread("lenna.png")  #读取照片 照片存储路径层跟py文件是同一个路径层
h,w =img.shape[:2]  #获取图片高和宽
img_gray=np.zeros([h,w],img.dtype) #创建一张新照片和当前图片一样大小的单通道图片

# 灰度化
for i in range(h):
    for j in range(w):
        m=img[i,j]
        img_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)

print("---m----")
print(m)
print("---image lenna----")
print(img)
print("---img_gray----")
print(img_gray)



plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')



#二值化 方法1
img_gray = rgb2gray(img)
rows,cols=img_gray.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i,j]<=0.5):
            img_gray[i,j]=0
        else:
            img_gray[i,j]=1

img_binary=img_gray
#二值化 方法2
#img_binary=np.where(img_gray>=0.5,1,0)
print("-----img_gray------")
print(img_gray)

print("-----image_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()



