import cv2
import numpy as np


#1.读取图片
org_img = cv2.imread("../../lenna.png")

#2.创建相同大小的单通道空图片
gray_img = np.zeros((org_img.shape[:2]),dtype=np.uint8)

#3.获取图片的长宽
h,w =org_img.shape[:2]

#4.循环原图片每个像素点位置，计算灰度值,计算公式：浮点算法：gray = R0.3 +G0.59 +B0.11
#  计算灰度值后再赋值给gray_img
for i in range(h):
    for j in range(w):
        gray_img[i,j] = org_img[i,j][0]*0.11 + org_img[i,j][1]*0.59 + org_img[i,j][2]*0.3

#分别显示原始图和灰度图
cv2.imshow("org_img",org_img)
cv2.imshow("gray_img",gray_img)

cv2.waitKey(0)
cv2.destroyAllWindows();























