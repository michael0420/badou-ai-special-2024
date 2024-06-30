import cv2
import numpy as np

img = cv2.imread('canny.png')

result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[430, 0], [0, 0], [430, 300], [0, 300]])
dst = np.float32([[0, 0], [0, 430], [300, 0], [300, 430]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (300, 430))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
