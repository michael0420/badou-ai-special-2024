

import numpy as np
import cv2

def function(img):
# 定义一个名为function的函数,接受一个参数img,表示输入的图像
    height, width, channels = img.shape
    # 获取输入图像的高度、宽度和通道数,并将它们分别赋值给变量height、width和channels
    emptyImage=np.zeros((800,800,channels),np.uint8)
    #创建一个大小为800x800像素的空图像,其通道数与输入图像相同
    sh=800/height
    #计算高度方向上的缩放比例sh,将目标高度800除以原始图像的高度height
    sw=800/width
    #计算宽度方向上的缩放比例sw,将目标宽度800除以原始图像的宽度width
    for i in range(800):
    # 开始一个嵌套的循环,外层循环变量i从0到799,表示目标图像的行索引
        for j in range(800):
        # 内层循环变量j从0到799,表示目标图像的列索引
            x=int(i/sh + 0.5)
            #int(),转为整型，使用向下取整。
            y=int(j/sw + 0.5)
            emptyImage[i,j]=img[x,y]
            # 将原始图像中位置(x,y)的像素值赋给目标图像中位置(i,j)的像素
    return emptyImage
    
# cv2.resize(img, (800,800,c),near/bin)

img=cv2.imread("lenna.png")
zoom=function(img)
#调用function函数,将读取的图像img作为参数传递,并将返回的结果赋值给变量zoom。

cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)

cv2.waitKey(0)


