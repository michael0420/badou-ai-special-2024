
'''
【第五周作业】
2.实现透视变换
'''
import cv2
import numpy as np
img = cv2.imread('photo1.jpg')
img_copy = img.copy()
# 获取原图上四个坐标点和对应到新图上的四个坐标点
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# 利用cv2封装的函数获取透视变换的矩阵
m = cv2.getPerspectiveTransform(src, dst)
print(m)
# 将原始图片上的每个坐标带入通用公式得到原始图片在新图上对应的每个坐标
# 生成出新图片
img_new = cv2.warpPerspective(img_copy, m, (337, 488))
cv2.imshow("img",img)
cv2.imshow("img_new",img_new)
cv2.waitKey(0)
