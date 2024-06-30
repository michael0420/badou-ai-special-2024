# canny边缘检测接口

import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
img_new = img.copy()

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

m = cv2.getPerspectiveTransform(src, dst)  # 生成3x3的透视变换矩阵
print('warpMatrix = \n', m)

img_new = cv2.warpPerspective(img_new, m, [337, 488])  # 被处理的图像、变换矩阵、变换后尺寸大小
cv2.imshow('original img', img)
cv2.imshow('perspective transformation', img_new)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

