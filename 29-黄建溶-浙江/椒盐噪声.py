import cv2
import random
def JyNoise(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])   # 确定需要添加噪声的点的数目
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)          # 随机确定需要椒盐的点
        randY=random.randint(0,src.shape[1]-1)
        if random.random()<=0.5:                   # 判断(椒盐化)
           NoiseImg[randX, randY]=0
        else:
            NoiseImg[randX, randY]=255
    return NoiseImg

img = cv2.imread('lenna.png',0)
img1 = JyNoise(img,0.2)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_JyNoiseNoise',img1)
cv2.waitKey(0)