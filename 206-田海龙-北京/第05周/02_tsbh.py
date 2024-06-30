
import cv2
import numpy as np

from utils import cv_imread, current_directory

img_path = current_directory + '\\img\\photo1.jpg'
img = cv_imread(img_path)

result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
# 更改原始坐标点，看实际效果
src = np.float32([[200, 141], [529, 280], [2, 602], [344, 742]])
dst = np.float32([[0, 0], [437, 0], [0, 588], [437, 588]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (437, 588))
# cv2.imshow("src", img)
# cv2.imshow("result", result)
# cv2.waitKey(0)

import matplotlib.pyplot as plt
plt.figure(1)
plt.imshow(img)
# plt.show()
plt.figure(2)
plt.imshow(result)
plt.show()
