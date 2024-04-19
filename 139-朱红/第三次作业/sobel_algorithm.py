import cv2
import numpy as np

# 加载图像
# flags:标志位
# cv2.IMREAD_COLOR：（默认参数）读入彩色图像忽略alpha通道，用1代替；
# cv2.IMREAD_GRAYSCALE：读入灰度图像，用0代替；
# cv2.IMREAD_UNCHANGED：读入完整图片包括alpha通道，用-1代替
img = cv2.imread("lenna.png", 0)

# 求x和y方向上的sobel算子
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # dx=1，dy=0，表示计算X方向的导数，检测出的是垂直方向上的边缘
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)  # dx=0，dy=1，表示计算Y方向的导数，检测出的是水平方向上的边缘

# 结果取绝对值（转换成uint8）
abs_sobel_x = cv2.convertScaleAbs(sobel_x)
abs_sobel_y = cv2.convertScaleAbs(sobel_y)

# 线性混合
add_sobel = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

# 显示垂直,水平和线性混合后的图像
cv2.imshow('sobel algorithm', np.hstack([abs_sobel_x, abs_sobel_y, add_sobel]))
# cv2.imshow('absX', abs_sobel_x)
# cv2.imshow('absY', abs_sobel_y)
# cv2.imshow('add', add_sobel)

cv2.waitKey(0)
cv2.destroyAllWindows()
