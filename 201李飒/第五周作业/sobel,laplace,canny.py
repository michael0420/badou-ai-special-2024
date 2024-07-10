import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sobel_x=cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
sobel_y=cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)

laplace = cv2.Laplacian(gray,cv2.CV_16S,ksize=3)

canny = cv2.Canny(gray,100,200)

plt.subplot(231), plt.imshow(gray, 'gray'), plt.title('original')
plt.subplot(232), plt.imshow(sobel_x, 'gray'), plt.title('sobel_x')
plt.subplot(233), plt.imshow(sobel_y, 'gray'), plt.title('sobel_y')
plt.subplot(234), plt.imshow(laplace, 'gray'), plt.title('laplace')
plt.subplot(235), plt.imshow(canny, 'gray'), plt.title('canny')
plt.show()