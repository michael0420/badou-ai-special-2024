import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.color import rgb2gray

#灰度图
def gray():
    for i in range(h):
        for j in range(w):
            mid=img[i,j]
            gray_img[i,j]=int(mid[0] * 0.11 + mid[1] * 0.59 + mid[2] * 0.3)

#二值图
def binary():
    for i in range(h):
        for j in range(w):
            mid=gray_img[i,j]/255
            if(mid>=0.5):
                binary_img[i,j]=0
            else:
                binary_img[i,j]=255

#红色加强
def red():
    for i in range(h):
        for j in range(w):
            red_img[i,j]=img[i,j]
            red_img[i,j,0]=np.where(red_img[i,j,2]*2>255,255,red_img[i,j,2]*2)   #0通道在cv2中虽然是bule，但是后续用了plt.imshow


img=cv2.imread("lenna.png")  #cv2读出来的是BGR
h,w=img.shape[:2]            #image为512*512*3
gray_img=np.zeros([h,w],img.dtype)     #三通道转单通道
binary_img=np.zeros([h,w],img.dtype)
red_img=np.zeros([h,w,3],img.dtype)
gray()
binary()
red()
cv2.imshow("image show gray", gray_img)
cv2.imshow("image show binary", binary_img)


#2*2画布
plt.subplot(221)             #2*2的，一共能放4个，1代表是第一个位置
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))    #plt.imshow不同于cv2.imshow,需要转换
plt.subplot(222)
plt.imshow(gray_img,cmap='gray')    #cmap是colormap的简称，用于指定渐变色
plt.subplot(223)
plt.imshow(binary_img,cmap='gray')
plt.subplot(224)
plt.imshow(red_img)
plt.show()

#简洁版
image=plt.imread("lenna.png")
#img_gray = rgb2gray(img)
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     #此处img_gray的值在0到1之间
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(121)
plt.imshow(img_gray,cmap='gray')
plt.subplot(122)
plt.imshow(img_binary,cmap='gray')
plt.show()

#plt.imshow方法首先将二维数组的值标准化为0到1之间的值，然后根据指定的渐变色依次赋予每个单元格对应的颜色，就形成了热图
#cv2.imshow接收两个参数，图像名以及图像矩阵
#np.where 函数是三元表达式 x if condition else y 的向量化版本
#np.where(condition,x,y) 当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y
