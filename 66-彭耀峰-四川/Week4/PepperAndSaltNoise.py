'''
实现椒盐噪声
'''

import cv2
import random

def PepAndSaltNoise(src,percent):
    NoiseImg = src
    NoiseNum = int(src.shape[0]*src.shape[1]*percent)
    for i in range(NoiseNum):
        randX = random.randint(0,src.shape[0] - 1)
        randY = random.randint(0,src.shape[1] - 1)
        #随机赋值为0或255
        if random.random() <= 0.5:
            NoiseImg[randX,randY] = 0
        elif random.random() > 0.5:
            NoiseImg[randX, randY] = 255
    return NoiseImg

img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
noiseImg = PepAndSaltNoise(img2,0.2)
cv2.imshow("PepperAndSalt img", noiseImg)
cv2.waitKey(0)