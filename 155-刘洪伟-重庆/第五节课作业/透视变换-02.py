# _*_ coding: UTF-8 _*_
# @Time: 2024/4/25 09:23
# @Author: iris
# @Email: liuhw0225@126.com
import cv2
import numpy as np

image = cv2.imread('../data/photo.jpg')

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(image, m, (337, 488))
cv2.imshow('result', result)
cv2.waitKey(0)
