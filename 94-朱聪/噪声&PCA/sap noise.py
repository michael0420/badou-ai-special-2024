import numpy as np
import cv2
import random
import copy


image = cv2.imread('../lenna.png', 0)

height, width = image.shape[0:2]

percent = 0.8

noise_img = copy.deepcopy(image)

noise_num = int(percent * height * width)

for i in range(noise_num):
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)

    random_num = random.random()

    if random_num < 0.5:
        noise_img[x, y] = 0
    else:
        noise_img[x, y] = 255


cv2.imshow('source', image)
cv2.imshow('noise', noise_img)

cv2.waitKey(0)

