import cv2
import numpy as np

img = cv2.imread('lenna.png',0)
# 等价于:
# original = cv2.imread('lenna.png') #默认为1：彩色读取
# img = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)

'''
# Sobel(src,ddepth,dx,dy,dst,ksize,scale,delta,borderType)
# 原图是unit8 无符号8位数
# cv2.CV_16S 防止阶段位数不够，使用有符号16位数
'''
x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)

'''
#在经过处理后，用convertScaleAbs()函数将其转回原来的uint8形式。
'''
abs_x = cv2.convertScaleAbs(x)
abs_y = cv2.convertScaleAbs(y)

'''
将两个方向组合起来
cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
gamma是加到最后结果上的一个值。
'''
result = cv2.addWeighted(abs_x,0.5,abs_y,0.5,0)

cv2.imshow("absX", abs_x)
cv2.imshow("absY", abs_y)

cv2.imshow("sobel result",result)
cv2.waitKey(0)
