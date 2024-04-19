import numpy as np
import matplotlib.pyplot as plt
import cv2


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb2gray(img):
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
    img_gray = img_gray / 255.0  # 归一化到0-1范围

    return img_gray


def binaryzation(img_gray):
    h, w = img_gray.shape[:2]
    img_binary = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            if img_gray[i, j] <= 0.5:
                img_binary[i, j] = 0
            else:
                img_binary[i, j] = 1
    return img_binary

img = cv2.imread("lenna.png")
img_rgb = bgr2rgb(img)
print("image show rgb: %s" % img_rgb)
img_gray = rgb2gray(img_rgb)
print("image show gray: %s" % img_gray)
img_binary = binaryzation(img_gray)
print("image show binary: %s" % img_binary)
"""
    plt方式灰度化和二值化
"""
plt.subplot(221)
plt.imshow(img_rgb)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
plt.subplot(223)
plt.imshow(img_binary, cmap='binary')
plt.show()
"""
    cv方式灰度化和二值化
"""
cv2.imshow("image show rgb", img)
# img_gray = (img_gray * 255).astype(np.uint8) # 灰度图像不需要转换为0-255范围
cv2.imshow("image show gray", img_gray)
img_binary = (img_binary * 255).astype(np.uint8) # 二值化图像需要转换为0-255范围
cv2.imshow("image show binary", img_binary)
cv2.waitKey(0)