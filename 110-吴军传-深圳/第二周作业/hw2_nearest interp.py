import cv2
import numpy as np


def funtion(Oldimage):
    height, width, channels = Oldimage.shape  # 三通道数不变
    EmptyImage = np.zeros((900, 900, channels), np.uint8)
    sh = 900 / height    #计算比例，
    sw = 900 / width     #计算比例
    for i in range(900):
        for j in range(900):
            x = int(i / sh + 0.5)# int(),转为整型，使用向下取整。 x、y表示原始图像坐标
            y = int(j / sw + 0.5)
            EmptyImage[i, j] = Oldimage[x, y] #计算出来的图像坐标赋值给新建的空图像
    return EmptyImage


Oldimage = cv2.imread("lenna.png")
Newimage = funtion(Oldimage)
print(Newimage)
print(Newimage.shape)
cv2.imshow("neighbor interpolation", Newimage)
cv2.imshow("image", Oldimage)
cv2.waitKey(0)
