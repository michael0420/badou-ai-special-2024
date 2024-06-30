# 椒盐噪声
import cv2
import random
from skimage import util


# 将原图转为椒盐噪声图片
def afterImg(src, rate):
    noiseImg = src
    noiseNum = int(rate * src.shape[0] * src.shape[1])

    for i in range(noiseNum):
        # random.randint生成随机整数
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5:
            noiseImg[randX, randY] = 0
        else:
            noiseImg[randX, randY] = 255
    return noiseImg


img = cv2.imread('lenna.png', 0)
img1 = afterImg(img, 0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用工具类转换椒盐噪声
img3 = util.random_noise(img, mode='s&p')

cv2.imshow('source', img2)
cv2.imshow('after', img1)
cv2.imshow('util_img', img3)
cv2.waitKey(0)
