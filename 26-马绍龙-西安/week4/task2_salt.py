import cv2 as cv
import random
import numpy as np


def salt_gray(img, percetage):
    h, w = img.shape
    result = img.copy()
    noise_num = int(h * w * percetage)
    for i in range(noise_num):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        tmp = random.random()
        if tmp < 0.5:
            result[x, y] = 0
        elif tmp > 0.5:
            result[x, y] = 255
    return result


def salt_color(img_color, percetage):
    row, col, channel = img_color.shape
    result_color = img_color.copy()
    noise_num = int(row * col * percetage)
    for c in range(channel):  # 对于彩色图像，表现为在单个像素BGR三个通道随机出现的255或0。所以分通道循环。
        for num in range(int(noise_num / channel)):  # 每个通道循环三分之一个点
            x = random.randint(0, row - 1)
            y = random.randint(0, col - 1)
            tmp = random.random()
            if tmp < 0.5:
                result_color[x, y, c] = 0
            elif tmp > 0.5:
                result_color[x, y, c] = 255
    return result_color


img = cv.imread('../week2/lenna.png', 0)
img_color = cv.imread('../week2/lenna.png', 1)
result = salt_gray(img, 0.01)
result_color = salt_color(img_color, 0.1)
cv.imshow('salt-compare-gray', np.hstack([img, result]))
cv.imshow('salt-compare-color', np.hstack([img_color, result_color]))
cv.waitKey(0)
