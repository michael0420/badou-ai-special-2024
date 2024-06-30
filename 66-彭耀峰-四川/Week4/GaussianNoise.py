'''
实现高斯噪声
'''

import cv2
import random

def GaussNoise(src,means,sigma,percent):
    NoiseImg = src
    NoiseNum = int(src.shape[0]*src.shape[1]*percent)
    for i in range(NoiseNum):
        randX = random.randint(0,src.shape[0] - 1)
        randY = random.randint(0,src.shape[1] - 1)
        #高斯随机数
        NoiseImg[randX,randY] = NoiseImg[randX,randY] + random.gauss(means,sigma)
        #将像素值缩放到0-255之间
        if NoiseImg[randX,randY] < 0:
            NoiseImg[randX,randY] = 0
        elif NoiseImg[randX,randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg

img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
noiseImg = GaussNoise(img2,2,4,0.8)
cv2.imshow("GaussianNose img", noiseImg)
cv2.waitKey(0)