# -*- coding: utf-8 -*-
"""
@author: zhjd

"""
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("alex.jpg", 1)

# Convert image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Apply Sobel operator for x and y derivatives
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # For x derivative
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # For y derivative

# Apply Laplace operator
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)

# Apply Canny edge detector
img_canny = cv2.Canny(img_gray, 100, 150)

# Plot images
plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")
plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("Sobel_x")
plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("Sobel_y")
plt.subplot(234), plt.imshow(img_laplace, "gray"), plt.title("Laplace")
plt.subplot(235), plt.imshow(img_canny, "gray"), plt.title("Canny")
plt.show()
