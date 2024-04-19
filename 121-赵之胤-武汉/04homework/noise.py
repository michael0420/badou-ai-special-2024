import cv2
import random
import numpy as np


def g_noise(img, mu, sigma, snr):
    noiseimg = img
    noisenum = int(snr*img.shape[0]*img.shape[1])
    for i in range(noisenum):
        rax = random.randint(0, img.shape[0]-1)
        ray = random.randint(0, img.shape[1]-1)

        noiseimg[rax, ray] = noiseimg[rax, ray] + random.gauss(mu, sigma)

        if noiseimg[rax, ray] < 0:
            noiseimg[rax, ray] = 0
        elif noiseimg[rax, ray] > 255:
            noiseimg[rax, ray] = 255
    return noiseimg


def jiao_noise(src, snr):
    res = src
    num = int(snr*src.shape[0]*src.shape[1])
    for i in range(num):
        x = random.randint(0, src.shape[0]-1)
        y = random.randint(0, src.shape[1]-1)

        if random.random() < 0.5:
            res[x, y] = 0
        else:
            res[x, y] = 255
    return res


img = cv2.imread("lenna.png", 0)
img1 = g_noise(img,2,4,0.8)
cv2.imshow('gaoso', img1)
img2 = jiao_noise(img, 0.2)
cv2.imshow('jioayan', img2)
img = cv2.imread("lenna.png")
img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', img3)
cv2.waitKey(0)
