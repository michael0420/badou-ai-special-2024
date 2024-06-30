import random
import cv2
import numpy as np


def slat_pepper_noise(image, ratio):
    h, w = image.shape[0:2]
    print(h, w)
    nosie_ratio = h * w * ratio

    for i in range(nosie_ratio.__int__()):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)

        if random.random() <= 0.5:
            image[y, x] = 255
        else:
            image[y, x] = 0
    return image


if __name__ == '__main__':
    source_img = cv2.imread('lenna.png', flags=0)
    cv2.imshow('source', source_img)
    dest_img = slat_pepper_noise(source_img, 0.5)
    cv2.imshow('dest',dest_img)
    cv2.waitKey(0)
