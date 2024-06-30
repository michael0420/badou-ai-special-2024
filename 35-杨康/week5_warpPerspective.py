import cv2
import numpy as np

"""
1.用4的点的图像坐标求出透视变换矩阵cv2.getPerspectiveTransform
2.进行透视变换cv2.warpPerspective
"""
img = cv2.imread('photo1.jpg')
result1 = img.copy()

#1.用4的点的图像坐标求出透视变换矩阵cv2.getPerspectiveTransform,图像坐标可以用画图或寻找顶点算法获得
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
warpPerspectiveMatrix = cv2.getPerspectiveTransform(src, dst)

#2.进行透视变换cv2.warpPerspective
result = cv2.warpPerspective(result1, warpPerspectiveMatrix, (337, 488))
cv2.imshow('src', img)
cv2.imshow('result', result)
cv2.waitKey(0)