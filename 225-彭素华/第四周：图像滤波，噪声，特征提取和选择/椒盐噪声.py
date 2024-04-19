import numpy as np
import cv2
from numpy import shape
import random


def saltpepper(src, percentage):
    NoiseImg = src
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        '''
        每次取一个随机点，把一张图片的像素用行和列表示的话，randX代表随机生成的行，randY代表随机生成的列。
        random.randint()生成随机数。
        '''
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        '''
        random.random()生成一个 0-1 之间的随机浮点数，随意渠道一个像素点有一半的可能是百点255，一半的可能是黑点0
        '''
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread('lenna.png', 0)
img1 = saltpepper(img, 0.8)
cv2.imshow("lenna_saltandpepper", img1)
cv2.waitKey(0)
