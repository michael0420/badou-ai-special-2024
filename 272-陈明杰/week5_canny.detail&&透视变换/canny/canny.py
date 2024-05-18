import cv2
import numpy as np

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是阈值1；
第三个参数是阈值2。
'''

# 以灰度形式读取图像
gray = cv2.imread("lenna.png",0)
img = cv2.Canny(gray,threshold1=100,threshold2=300)
cv2.imshow("src",gray)
cv2.imshow("img",img)
cv2.waitKey(0)