import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

result = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [666, 0], [0, 999], [666, 999]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result, m, (666, 999))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)