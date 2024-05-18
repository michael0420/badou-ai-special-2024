import cv2
import numpy as np

cv__imread = cv2.imread("photo1.jpg")

imread_copy = cv__imread.copy()
'''
需求需告诉我输入四个点的坐标和输出八个点的坐标
'''
original = np.float32([[207, 151], [517, 285], [17, 601], [343, 732]])
target = np.float32([[0, 0], [337, 0], [0, 485], [337, 485]])
print(cv__imread.shape)
# 进行透视转换
toushi_target = cv2.getPerspectiveTransform(original, target)
print("warpMatrix:")
print(toushi_target)
perspective = cv2.warpPerspective(imread_copy, toushi_target, (337, 485))
cv2.imshow("src", cv__imread)
cv2.imshow("result", perspective)
cv2.waitKey(0)
