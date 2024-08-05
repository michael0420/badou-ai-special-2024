# -*- coding:utf-8 -*-

__author__ = 'Young'

import cv2
import matplotlib.pyplot as plt

import myimage
import numpy as np
import PerspectiveTranformationUtil as PTUtil

"""
1.实现canny detail 
2.实现透视变换
"""

"""
1.实现canny detail 
"""
lenna = myimage.MyImage('../Image/lenna.png')
lenna.canny_detail()
plt.figure(1)
plt.imshow(lenna.gray_image.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.figure(2)
plt.imshow(lenna.sobel_image.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.figure(3)
plt.imshow(lenna.nms_image.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.figure(4)
plt.imshow(lenna.canny_image.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.show()

"""
2.实现透视变换
"""
# 2.1 变换
photo1 = cv2.imread('../Image/photo1.jpg')
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
pt_photo1 = PTUtil.perspective_transformation(photo1, src, dst, (337, 488))
cv2.imshow('photo 1', photo1)
cv2.imshow('pt photo 1', pt_photo1)

# 2.2 寻找目标图像坐标
v_photo1 = PTUtil.find_vertices(photo1)
cv2.imshow('find src', v_photo1)

# 2.3 warpMatrix
src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
src = np.array(src)

dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
dst = np.array(dst)

warp_matrix = PTUtil.warp_perspective_matrix(src, dst)
print(warp_matrix)

cv2.waitKey(0)
cv2.destroyAllWindows()
