import cv2
import random
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])   # 确定需要添加噪声的点的数目
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)          # 随机确定需要高斯化的点
        randY=random.randint(0,src.shape[1]-1)          # 随机确定需要高斯化的点
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)   #高斯化
        if NoiseImg[randX, randY] <0:                   # 判断
           NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg

img = cv2.imread('lenna.png',0)
img1 = GaussianNoise(img,2,4,0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)





