import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])  # 原图像坐标点
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])  # 目标图像坐标点
print(img.shape)  # (960, 540, 3)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
'''
warpMatrix:
[[ 8.92263779e-01  3.76733596e-01 -2.41585375e+02]
 [-4.08140258e-01  9.44205073e-01 -5.80899328e+01]
 [-8.53836442e-05  5.16464182e-05  1.00000000e+00]]
'''
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
