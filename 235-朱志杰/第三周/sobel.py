import cv2
import numpy as np

img = cv2.imread("lenna.png")

# 使用16位计算图片水平、垂直方向梯度，因原图8位不够进行计算。
# cv2.Sobel(src, ddepth, dx, dy, ksize) ddepth为图像位数，dx、dy为水平、垂直方向求导,ksize卷积核大小
x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

# 取绝对值，因16计算存在负数
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

# 合并水平、垂直方向图片。alpha、beta为权重，gamma为亮度调节
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("Result", dst)

cv2.waitKey(0)
# 关闭所有窗口
cv2.destroyAllWindows()