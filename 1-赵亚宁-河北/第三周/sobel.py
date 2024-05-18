import cv2
import numpy as np

# 读取图片
image = cv2.imread('lenna.png')

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Sobel算子进行边缘检测
sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=5)  # 水平方向
sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=5)  # 垂直方向

# 结合水平和垂直方向的梯度
sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

# 显示结果
cv2.imshow('Sobel Edge Detection', sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('Sobel_output.jpg', sobel)