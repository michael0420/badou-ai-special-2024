import numpy
import cv2
import random


def pepper_noise(src, percentage):
    NoiseImage = src.copy()
    NoiseNum = int(percentage*NoiseImage.shape[0]*NoiseImage.shape[1])
    for i in range(NoiseNum):
        rand_x = random.randint(0, NoiseImage.shape[0]-1)
        rand_y = random.randint(0, NoiseImage.shape[1]-1)
    # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5:
            NoiseImage[rand_x, rand_y] = 0
        else:
            NoiseImage[rand_x, rand_y] = 255
    return NoiseImage


img = cv2.imread("lenna.png",0)
img1 = pepper_noise(img, 0.1)
cv2.imshow('source', img)
cv2.imshow('lenna_pepper_noise', img1)
cv2.waitKey(0)

