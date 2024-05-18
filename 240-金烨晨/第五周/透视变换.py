import cv2
import numpy as np

img = cv2.imread("photo1.jpg")
result1 = img.copy()

# 定义原始图像和目标图像上的四个关键点
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])   # 原始图像的四个顶点坐标
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])          # 目标图像的四个顶点坐标

# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)               # 得到透视变换矩阵
result = cv2.warpPerspective(result1, m, [337, 488])    # 应用透视变换

# 显示原始图像和变换后的结果图像
cv2.imshow("src", img)
cv2.imshow("dst", result)
cv2.waitKey()