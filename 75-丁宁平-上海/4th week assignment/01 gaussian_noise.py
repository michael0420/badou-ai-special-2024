import numpy as np
import cv2
import random

def GaussianNoise(src,means,sigma,percentage):
    Noiseimg = src.copy()
    Noisenum = int(Noiseimg.shape[0]*Noiseimg.shape[1]*percentage)
    for i in range(Noisenum):
        random_X = random.randint(0,Noiseimg.shape[0]-1)
        random_y = random.randint(0,Noiseimg.shape[1]-1)
        Noiseimg[random_X,random_y] += random.gauss(means,sigma)
        if Noiseimg[random_X,random_y] < 0:
            Noiseimg[random_X, random_y] = 0
        elif Noiseimg[random_X,random_y] > 255:
            Noiseimg[random_X,random_X] = 255
    return Noiseimg

img = cv2.imread("Grace.jpg",0)

GN_img = GaussianNoise(img,8,8,0.9)
cv2.imshow('source',img)
cv2.imshow("GaussianNoise",GN_img)
cv2.waitKey(0)



