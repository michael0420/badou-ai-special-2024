"""
@author: huangyunxuan

透视转换

"""

import cv2

import numpy as np


img = cv2.imread("photo1.jpg")
img2 = img.copy()

src=np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst=np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
mat = cv2.getPerspectiveTransform(src,dst)
print("转换矩阵位：")
print(mat)
img_r = cv2.warpPerspective(img2, mat, (450,500))

cv2.imshow("img",img)
cv2.imshow("PT",img_r)
cv2.waitKey(0)