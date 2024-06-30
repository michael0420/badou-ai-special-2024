import cv2
import numpy as np


def function(img):
    height, width, channels = img.shape  # 原图的 height、width、channels(3通道)
    emptyImage = np.zeros((800, 800, channels), np.uint8)  # 放大成 800X800 通道数不变:3通道
    sh = 800 / height  # sh 放大倍数: 800/512
    sw = 800 / width  # sw 放大倍数: 800/512
    for i in range(800):  # i 为放大后图像的坐标
        for j in range(800):  # j 为放大后图像的坐标
            x = int(i / sh + 0.5)  # x 原始图像的坐标    int(),转为整型，使用向下取整
            y = int(j / sw + 0.5)  # y 原始图像的坐标
            """
            例：     2.2    2.5    2.7
              +0.5  2.7    3      3.2   int(放大后坐标/放大比例+0.5）效果等于四舍五入   
                    2      3      3
            """
            emptyImage[i, j] = img[x, y]  # 经过计算后原始图像上的坐标 赋值给 放大图像的坐标
    return emptyImage


# cv2.resize(img, (800,800,c),near/bin)  直接调用api
img = cv2.imread("D:\cv_workspace\picture\lenna.png")  # 读取原图 512X512  shape=(512,512,3)  dtype=unit8
zoom = function(img)  # 定义function函数实现上采样
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
