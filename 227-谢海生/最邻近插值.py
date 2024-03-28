import cv2
import numpy as np
#定义函数function放入参数img
def function(img):
    #获取图像img的形状，其中高度和宽度都发生变化，但是通道数还是原来的通道数
    height,width,channels = img.shape
    #建立一个新的图像变量
    emptyImage=np.zeros((800,800,channels),np.uint8)
    #利用现在的像素除以高度和宽度求出缩放比例
    sh=800/height
    sw=800/width
    #for循环，i和j取0到799是坐标，其中x和y是坐标，转为整型之后，向下求取整数，所以加0.5便于计算，最临近插值新的坐标离谁近就插入到那个区域
    for i in range(800):
        for j in range(800):
            #类似i+u j+v
            x=int(i/sh + 0.5)
            y=int(j/sw + 0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage
#读取图片
img=cv2.imread("C:/Users/86188/Pictures/lenna.png")
#定义的函数存储到变量zoom中
zoom=function(img)
print(zoom)
print(zoom.shape)
#显示缩放后的图像
cv2.imshow("nearest interp",zoom)
#显示原图像
cv2.imshow("image",img)
cv2.waitKey(0)