
"""
彩色图像灰度化
"""

import cv2
import numpy as np
from skimage.color import rgb2gray

# 1.手动实现
def rgb2grayNew(img):
    h, w = img.shape[:2]  # 取前两个值，图片的高high和宽wide
    img_gray = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
    return img_gray


img = cv2.imread("lenna.png")
img_gray1 = rgb2grayNew(img)
cv2.imshow("img show gray1",img_gray1)
cv2.waitKey(0)


# 2.接口实现
# img_gray2 = rgb2gray(img)
# cv2.imshow("img show gray2",img_gray2)
# cv2.waitKey(0)

