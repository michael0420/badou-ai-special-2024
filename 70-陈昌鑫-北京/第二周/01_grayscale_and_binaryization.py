import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

def grayscale_1(src_img):
    gray_img = np.zeros([src_img.shape[0], src_img.shape[1]], np.uint8)
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            src = src_img[i, j]
            gray_img[i, j] = (0.11 * src[0] + 0.59 * src[1] + 0.3 * src[2])
    return gray_img

def grayscale_2(src_img):
    gray_img = np.zeros([src_img.shape[0], src_img.shape[1]], np.float32)
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            src = src_img[i, j]
            gray_img[i, j] = (0.11 * src[0] + 0.59 * src[1] + 0.3 * src[2]) / 255
    return gray_img

def binaryization(src_img):
    binary_img = np.zeros((src_img.shape[0], src_img.shape[1]), np.float32)
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            binary_img[i, j] = round(src_img[i, j])
    return binary_img

src_img = cv2.imread("lenna.png")
gray_img_1 = grayscale_1(src_img)

plt.subplot(321)
plt.imshow(gray_img_1, cmap='gray')

gray_img_2 = rgb2gray(src_img)
plt.subplot(322)
plt.imshow(gray_img_2, cmap='gray')

gray_img_3 = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
plt.subplot(323)
plt.imshow(gray_img_3, cmap='gray')

gray_img_4 = grayscale_2(src_img)
plt.subplot(324)
plt.imshow(gray_img_4, cmap='gray')

binary_img_1 = binaryization(gray_img_4)
plt.subplot(325)
plt.imshow(binary_img_1, cmap='gray')


binary_img_2 = np.where(gray_img_2 >= 0.5, 1, 0)
plt.subplot(326)
plt.imshow(binary_img_2, cmap='gray')

plt.show()

