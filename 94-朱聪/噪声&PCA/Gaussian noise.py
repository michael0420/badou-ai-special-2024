
import numpy as np
import cv2
import random
import copy

image = cv2.imread('../lenna.png', 0) # 0是读取灰度图，默认为1，读取彩色图(忽略透明度)

height, width = image.shape

percetage = 0.8

noise_num = int(percetage * height * width)

noise_img = copy.deepcopy(image)

for i in range(noise_num):
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)

    noise_img[x, y] = image[x, y] + random.gauss(2, 4)

    if noise_img[x, y] > 255:
        noise_img[x, y] = 255
    elif noise_img[x, y] < 0:
        noise_img[x, y] = 0


cv2.imshow('source', image)
cv2.imshow('noise', noise_img)
cv2.waitKey(0)