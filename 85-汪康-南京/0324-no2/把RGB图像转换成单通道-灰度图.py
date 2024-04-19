# -*- coding: utf-8 -*-
'''@Time: 2024/3/24 17:23
把RGB图像转换成单通道-灰度图
注意cv2读取的是BGR
'''
import numpy as np
import cv2

#方法1：用cv2

image = cv2.imread("../lenna.png")
# print(f'image:{image}')
h,w = image.shape[:2]
GreyImage = np.zeros([h,w],image.dtype)
for i in range(h):
    for j in range(w):
        B,G,R = image[i,j][0],image[i,j][1],image[i,j][2]
        GreyImage[i,j] = int(R*0.3+G*0.59+B*0.11)
    # print(GreyImage)

print(GreyImage)
cv2.imshow("rgb",image)
cv2.imshow("grey",GreyImage)
cv2.waitKey(0)
TwoImage = GreyImage
h2, w2 = TwoImage.shape
for i in range(h2):
    for j in range(w2):
        if TwoImage[i, j] /255 <= 0.5:
            TwoImage[i, j] = 0
        else:
            TwoImage[i, j] = 255 #cv2.imshow只能显示0-255数值
print(TwoImage)
cv2.imshow("two",TwoImage)
cv2.waitKey(0)

