import cv2
import numpy as np

img = cv2.imread("lenna.png", 0)

'''
Sobel函数求完导数后会有负值，还有会大于255的值。
而原图像是uint8，即8位无符号数(范围在[0,255])，所以Sobel建立的图像位数不够，会有截断。
因此要使用16位有符号的数据类型，即cv2.CV_16S。
'''

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
#水平梯度（竖向线条更明显）
absX = cv2.convertScaleAbs(x)
#垂直梯度（横向线条更明显）
absY = cv2.convertScaleAbs(y)

'''
由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
。其函数原型为：
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
注：
src1：第一张输入图像。
alpha：第一个输入图像的权重。
src2：第二张输入图像。
beta：第二个输入图像的权重。
gamma：加权和基础上添加的一个常数值。
dst：输出图像，默认情况下由函数分配内存，如果不为空，则直接将结果写入该矩阵中。
dtype：输出图像的类型，默认值为 -1 表示与输入图像相同的数据类型。
'''
#权重叠加
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)
cv2.imshow("Result", dst)
cv2.waitKey(0)

