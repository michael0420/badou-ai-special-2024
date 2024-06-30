import numpy as np
import cv2
import random

def Pp_Salt_Noise(src,percentage):
    Noiseimg = src.copy()
    Noisenum = int(percentage*Noiseimg.shape[0]*Noiseimg.shape[1])
    for i in range(Noisenum):
        rand_X = random.randint(0,Noiseimg.shape[0]-1)
        rand_y = random.randint(0, Noiseimg.shape[1]-1)
        if random.random() <= 0.5:
            Noiseimg[rand_X,rand_y] = 0
        else:
            Noiseimg[rand_X,rand_y] = 255

    return Noiseimg

img = cv2.imread("Grace.jpg")
Gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
Noise_img = Pp_Salt_Noise(Gray_img,0.2)
cv2.imshow("Source",Gray_img)
cv2.imshow('Pepper_Salt_Noise',Noise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()