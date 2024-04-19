# -*- coding: utf8 -*-

import grayscale
import cv2
import matplotlib.pyplot as plt

# Self implementation

img_gray = grayscale.convert_to_gray_image("lenna.png")
print('image show gray:%s' % img_gray)
cv2.imshow('image show gray', img_gray)
# cv2.waitKey(0)


# Use matplotlib.pyplot
# plt.subplot(221)

img_gray = grayscale.convert_to_gray_image_pyplot("lenna.png")
plt.imshow(img_gray, cmap='gray')
plt.show()


# Use cv2
img_gray = grayscale.convert_to_gray_image_cv2("lenna.png")
cv2.imshow('lenna gray image', img_gray)
cv2.waitKey(0)
