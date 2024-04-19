import cv2
import numpy as np

img = cv2.imread("lenna.png", 0)  #0:  gray
# 更加便捷，原本两步：
# img1 = cv2.imread('lenna.png') #默认为1：彩色读取
# img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

# Sobel(src,ddepth,dx,dy,dst,ksize,scale,delta,borderType)
# ddepth:图像深度——数据类型。 16s:16位有符号。 16U：16位无符号 32，64F：浮点
# dx = 1, dy = 0 : x方向一阶导。 dx = 0, dy = 1 : y方向一阶导。
# ksize: 算子大小。 默认为3。

# ___________________________________________
# Sobel函数求完导数后会有负值，还有会大于255的值。
# 原图像uint8，无符号8位数。(范围在[0,255])，Sobel建立的图像位数不够，会有截断。
# cv2.CV_16S：防止位数不够。

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

#经过处理后，用convertScaleAbs()函数将其转回原来的uint8形式.
abs_x = cv2.convertScaleAbs(x)
abs_y = cv2.convertScaleAbs(y)


# 将两个方向组合起来:
# cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
# 其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
# gamma是加到最后结果上的一个值。

dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

cv2.imshow("absX", abs_x)
cv2.imshow("absY", abs_y)

cv2.imshow("Result", dst)

cv2.waitKey(0)
# cv2.destroyAllWindows()