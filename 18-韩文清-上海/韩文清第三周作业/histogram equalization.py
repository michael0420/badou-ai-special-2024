import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取灰度图像
img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_GRAYSCALE 表示以灰度模式读取图像。

# 灰度图像直方图均衡化
equ = cv2.equalizeHist(img)

# 绘制均衡化后的直方图
plt.figure()
plt.hist(equ.ravel(), 256) #equ 是经过直方图均衡化处理后的灰度图像，.ravel() 函数用于将图像展平为一维数组，以便直方图函数进行处理。256表示直方图的 bin 的数量.
plt.title('Equalized Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# 显示均衡化后的直方图
plt.show()

# 显示原始灰度图像和均衡化后的图像
cv2.imshow("Original Image", img)
cv2.imshow("Equalized Image", equ)

# 等待用户按下任意键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
# 彩色图像直方图均衡化
# 读取彩色图像
img = cv2.imread("lenna.png", cv2.IMREAD_COLOR)

# 将彩色图像转换为 YUV 颜色空间
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# 对 Y 通道进行直方图均衡化处理
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# 将处理后的图像转换回 BGR 颜色空间
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# 显示原始彩色图像和直方图均衡化后的图像
cv2.imshow("Original Image", img)
cv2.imshow("Equalized Image", img_output)

# 等待用户按下任意键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''对 Y 通道进行均衡化处理而不是对彩色通道进行处理是因为 YUV 颜色空间中的 Y 通道代表了图像的亮度信息，
而 U 和 V 通道代表了图像的色度信息。
在彩色图像中，亮度（Y）与颜色（U 和 V）是分开存储的。
这意味着对 Y 通道进行直方图均衡化只会调整图像的亮度信息，而不会改变图像的颜色。
这样做的好处是可以增强图像的对比度，同时保持图像的自然色彩。
如果对整个彩色图像的 RGB 通道进行直方图均衡化处理，会导致图像的颜色失真，
因为 RGB 通道之间的相关性会被破坏，从而使得均衡化后的图像看起来不自然。'''