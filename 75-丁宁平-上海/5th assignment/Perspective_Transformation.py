# coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('photo1.jpg')
temp = img.copy()
plt.imshow(temp) # 查看顶点坐标，有xy坐标
plt.show()

# 输入是图像对应顶点坐标
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

# 生成透视变换矩阵；进行透视变换
pt = cv2.getPerspectiveTransform(src, dst)
print('warpMatrix：', pt, sep='\n')
rs = cv2.warpPerspective(temp, pt, (337, 488))

cv2.imshow('src', img)
cv2.imshow('result', rs)
cv2.waitKey(0)
cv2.destroyAllWindows()
