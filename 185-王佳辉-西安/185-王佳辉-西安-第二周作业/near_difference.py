import cv2
import numpy as np

def function(img):
    h,w,channels=img.shape #h：高，w：宽，h和w相当于一个坐标，channel：通道，当前lenna.png图片的通道数
    emptyImage = np.zeros((800,800,channels),np.uint8)#np.uint8:创建指定维度和数据类型未初始化的数组 empty
    sh=800/h #h的缩放或放大后的比例值
    sw=800/w #w的缩放或放大后的比例值
    for i in range(800):
        for j in range(800):
            x=int(i/sh+0.5)#计算当前坐标根据sh等比放大或缩小的值，+0.5的原因是转化为int是向下取整，放止取数与当前值差距过大
            y=int(j/sw +0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage


img=cv2.imread("img/lenna.png")

zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("show nearest  interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.imwrite("img/nearest_lenna.png",zoom)







