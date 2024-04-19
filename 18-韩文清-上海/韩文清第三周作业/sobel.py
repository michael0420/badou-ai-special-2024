import cv2
import numpy as np

import cv2

# 读取图像
img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)

# 使用Sobel算子进行水平方向的边缘检测
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  #64位浮点型，用于保存可能为负值的梯度值,Sobel算子的大小ksize=3。
# 使用Sobel算子进行垂直方向的边缘检测
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# 计算边缘强度
edge_strength = cv2.magnitude(sobelx, sobely) #使用cv2.magnitude()函数计算水平方向和垂直方向梯度的幅值，得到边缘强度图像。

# 显示边缘强度
cv2.imshow("Edge Strength", edge_strength.astype('uint8'))

# 等待用户按下任意键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
