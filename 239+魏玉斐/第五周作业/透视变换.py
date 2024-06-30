import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('photo1.jpg')

result = img.copy()
# 获取图像的四个顶点坐标
# result1 = plt.imread('photo1.jpg')
# plt.imshow(result)
# plt.show()
'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
利用已知坐标点的坐标和目标坐标点的坐标，可以计算出透视变换矩阵。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
# 利用透视矩阵做转换
result = cv2.warpPerspective(result, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
