import cv2
import random
from numpy import shape

def gaussnoise(img, mu, sigma, percent):
    noiseImg = img
    noiseNum = int(img.shape[0] * img.shape[1] * percent)
    for i in range(noiseNum):
        randomX = int(random.randint(0, img.shape[0]-1))
        randomY = int(random.randint(0, img.shape[1]-1))

        noiseImg[randomX, randomY] = noiseImg[randomX, randomY] + random.gauss(mu, sigma)
        if noiseImg[randomX, randomY] > 255:
            noiseImg[randomX, randomY] = 255
        elif noiseImg[randomX, randomY] < 0:
            noiseImg[randomX, randomY] = 0
    return noiseImg

def saltpeppernoise(img, percent):
    noiseImg = img
    noiseNum = int(img.shape[0] * img.shape[1] * percent)
    for i in range(noiseNum):
        randomX = int(random.randint(0, img.shape[0] - 1))
        randomY = int(random.randint(0, img.shape[1] - 1))
        randomnum = random.random()
        if randomnum < 0.5:
            noiseImg[randomX, randomY] = 0
        else:
            noiseImg[randomX, randomY] = 255
    return noiseImg
if __name__ == '__main__':
    imgsrc = cv2.imread('300.jpg', 0)
    img = imgsrc.copy()
    img1 = gaussnoise(img, -10, 4, 1)
    img = imgsrc.copy()
    img2 = saltpeppernoise(img, 0.5)
    cv2.imshow('source', imgsrc)
    cv2.imshow('gaussnoise', img1)
    cv2.imshow('saltpeppernoise', img2)
    cv2.waitKey(0)