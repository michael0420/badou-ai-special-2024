# 1.实现灰度化和二值化

import cv2
import numpy as np

# 方法一： 直接使用包中工具

# 读取图形
img = cv2.imread('lenna.png')
# 把图像转为灰度值
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 把灰度图二值化
ret,binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

print(ret)    # 打印阈值,因为threshold方法有2个返回值,分别为处理后的图形 和阈值.


# 显示图像
cv2.imshow("gray",img)
cv2.imshow('binary',binary)
cv2.waitKey(0)


# 方法二： 使用自定义方法计算

def gray(image):
    height,width = image.shape[0:2]
    gray = np.zeros((height,width,1),np.uint8)
    for i in range(height):
        for j in range(width):
            gray[i,j] = image[i][j][0]*0.11+image[i][j][1]*0.59+image[i][j][2]*0.3   #BGR排序，  浮点算法为： R*0.3+G*0.59+B*0.11
            # gray[i,j] = (float(image[i][j][0]) + float(image[i][j][1]) + float(image[i][j][2])) / 3    #平均值法
            # gray[i,j] =image[i][j][1]   #仅仅取绿色G
    return  gray

def binary(image):
    height,width = image.shape[0:2]
    gray_result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = np.zeros((height,width,1),np.uint8)
    for i in range(height):
        for j in range(width):
            if gray_result[i][j] < 127:
                binary[i][j] = 0
            else:
                binary[i][j] = 255
    return binary

img = cv2.imread('lenna.png')
gray_result = gray(img)
binary_reult = binary(img)
cv2.imshow("gray_manual",gray_result)
cv2.imshow("binary_manual",binary_reult)
cv2.waitKey(0)
