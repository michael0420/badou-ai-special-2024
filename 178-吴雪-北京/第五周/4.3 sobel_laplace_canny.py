"""
sobel and laplace and canny
invocation interface
"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('E:/Desktop/jianli/lenna.png', 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Sobel算子
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)
# Laplace算子
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
img_laplace = cv2.convertScaleAbs(img_laplace)
img_canny = cv2.Canny(img_gray, 100, 150)
# 显示
plt.subplot(231), plt.imshow(img_gray, cmap='gray'), plt.title('Original')
plt.subplot(232), plt.imshow(img_sobel_x, cmap='gray'), plt.title('Sobel_x')
plt.subplot(233), plt.imshow(img_sobel_y, cmap='gray'), plt.title('Sobel_y')
plt.subplot(234), plt.imshow(img_laplace, cmap='gray'), plt.title('Laplace')
plt.subplot(235), plt.imshow(img_canny, cmap='gray'), plt.title('Canny')
plt.subplot(236), plt.imshow(img_sobel, cmap='gray'), plt.title('Sobel')
plt.show()
