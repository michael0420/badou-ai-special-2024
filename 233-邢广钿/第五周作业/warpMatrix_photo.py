import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

result3 = img.copy()

#原坐标
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
#目标坐标
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)
# 生成透视变换矩阵
m = cv2.getPerspectiveTransform(src, dst)
# 透视变换
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)