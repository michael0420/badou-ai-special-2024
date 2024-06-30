import cv2
import random

def jiaoyan(src,percentage):
    noise_image = src
    noise_num = int(percentage * src.shape[0] * src.shape[1])
    for i in range(noise_num):
        randX = random.randint(0,src.shape[0] - 1)
        randY = random.randint(0,src.shape[1] - 1)
        if random.random() <= 0.5:
            noise_image[randX,randY] = 0
        else:
            noise_image[randX,randY] = 255
    return noise_image

imag1 = cv2.imread('../week5/lenna.png', 0)
image2 = jiaoyan(imag1,0.8)
cv2.imshow('image1',imag1)
cv2.imshow('image2',image2)
cv2.waitKey(0)