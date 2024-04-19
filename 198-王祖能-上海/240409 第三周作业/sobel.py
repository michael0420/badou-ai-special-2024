import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("city.jpg", 0)  # 原图为uint8,求导会有负值及超界255的值
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 防止超界阶段，计算深度改为16位有符号cv2.CV_16S，dx=1对x求导, dy=0对y不求导
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)  # 要用convertScaleAbs()转回uint8返回uint8类型的图片，否则灰色窗口
absY = cv2.convertScaleAbs(y)

x1 = cv2.Sobel(img, cv2.CV_16S, 2, 0)
y1 = cv2.Sobel(img, cv2.CV_16S, 1, 1)
absX1 = cv2.convertScaleAbs(x1)
absY1 = cv2.convertScaleAbs(y1)
'''
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。
'''
cv2.imshow('horizontal_grad', np.hstack([absX, absX1]))  # 水平梯度求导，竖向边缘检测，遇到边缘处值突变
cv2.imshow('vertical_grad', np.hstack([absY, absY1]))  # 垂直梯度矩阵，水平边缘检测
cv2.waitKey()
'''
将两方向组合，dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
gamma是加到最后结果上的一个值。
'''
merge1 = cv2.addWeighted(absX, 0.8, absY, 0.6, 100)
merge2 = cv2.addWeighted(absX, 0.5, absY, 0.5, -25)
# print(merge1 == merge2)
cv2.imshow('merge', np.hstack([merge1, merge2]))
cv2.waitKey()