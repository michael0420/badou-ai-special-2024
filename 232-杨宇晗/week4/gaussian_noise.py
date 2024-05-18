"""
@author: Hanley-Yang

高斯噪声
"""
import cv2
import random


def GaussianNoise(src, means, sigma, percentage):
    noiseImg = src
    noiseNum = int(percentage * src.shape[0] * src.shape[1])
    # 根据需要添加噪声的数量遍历，每次取随机像素点(randX,randY)
    for i in range(noiseNum):
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        # 在原有像素灰度值上加上随机数
        noiseImg[randX, randY] = noiseImg[randX, randY] + random.gauss(means, sigma)
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if noiseImg[randX, randY] < 0:
            noiseImg[randX, randY] = 0
        elif noiseImg[randX, randY] > 255:
            noiseImg[randX, randY] = 255
    return noiseImg
srcImg = cv2.imread('shangri-la.jpg',0)
gaussiamImg = GaussianNoise(srcImg, 2, 4, 0.8)
srcImg = cv2.imread('shangri-la.jpg')
srcImg_gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)

cv2.imwrite('gaussian_noise.jpg',gaussiamImg)

cv2.imshow('source', srcImg_gray)
cv2.imshow('GaussianNoise', gaussiamImg)
cv2.waitKey(0)






