import random

import cv2


def gn(src, means, sigma, percetage):
    ni = src
    nn = int(percetage * src.shape[0] * src.shape[1])
    for i in range(nn):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        ni[randX, randY] = ni[randX, randY] + random.gauss(means, sigma)
        if ni[randX, randY] < 0:
            ni[randX, randY] = 0
        elif ni[randX, randY] > 255:
            ni[randX, randY] = 255
    return ni

if __name__ == '__main__':
    img = cv2.imread('lenna.png',0)
    gauss = gn(img,2,4,0.9)
    img = cv2.imread('lenna.png')
    origin = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('source',origin)
    cv2.imshow('gauss',gauss)
    cv2.waitKey(0)