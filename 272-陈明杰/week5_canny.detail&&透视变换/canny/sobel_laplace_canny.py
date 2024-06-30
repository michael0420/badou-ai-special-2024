import cv2
from matplotlib import pyplot as plt
import numpy as np

# 读取彩色图
img = cv2.imread("lenna.png")
# 变成灰度图
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sobel边缘提取
sobel_x=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
sobel_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
# cv2.imshow("sobel_x",sobel_x)
# cv2.imshow("sobel_y",sobel_y)
# cv2.waitKey(0)

# laplace边缘提取
laplace=cv2.Laplacian(gray,cv2.CV_64F,ksize=3)
# cv2.imshow("laplace",laplace)
# cv2.waitKey(0)

# canny边缘提取
canny=cv2.Canny(gray,100,300)
# cv2.imshow("canny",canny)
# cv2.waitKey(0)

# 构建一个2行3列的图像框，并把接下来的图片放到第一个位置
plt.subplot(231),plt.imshow(gray,cmap='gray'),plt.title('gray')
plt.subplot(232),plt.imshow(sobel_x,cmap='gray'),plt.title('sobel_x')
plt.subplot(233),plt.imshow(sobel_y,cmap='gray'),plt.title('sobel_y')
plt.subplot(234),plt.imshow(laplace,cmap='gray'),plt.title('laplace')
plt.subplot(235),plt.imshow(canny,cmap='gray'),plt.title('canny')
plt.subplot(236),plt.imshow(img,cmap='gray'),plt.title('img')

plt.show()
