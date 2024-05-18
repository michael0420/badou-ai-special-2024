import random

import cv2


def pepperSalt(src, percetage):
    ni = src
    nn = int(percetage * src.shape[0] * src.shape[1])
    for i in range(nn):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.random() < 0.5:
            ni[randX, randY] = 0
        else:
            ni[randX, randY] = 255
    return ni


if __name__ == '__main__':
    img = cv2.imread('lenna.png', 0)
    gauss = pepperSalt(img,0.001)
    img = cv2.imread('lenna.png')
    origin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('source', origin)
    cv2.imshow('pepper salt', gauss)
    cv2.waitKey(0)
