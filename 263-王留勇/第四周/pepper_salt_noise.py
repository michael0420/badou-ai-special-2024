"""
椒盐噪声
"""

import cv2
import random

def pepper_salt_noise(src, percetage):
    noiseImg = src
    height, width = noiseImg.shape[:2]
    noiseNum = int(percetage * height * width)

    for i in range(noiseNum):
        randX = random.randint(0, height - 1)
        randY = random.randint(0, width - 1)
        if random.random() <= 0.5:
            noiseImg[randX, randY] = 0
        else:
            noiseImg[randX, randY] = 255
    return noiseImg


if __name__ == '__main__':
    img = cv2.imread('lenna.png', 0)
    print(img)
    cv2.imshow('source', img)
    noiseImg = pepper_salt_noise(img, 0.3)
    cv2.imshow('pepper_salt_img', noiseImg)
    cv2.waitKey(0)

