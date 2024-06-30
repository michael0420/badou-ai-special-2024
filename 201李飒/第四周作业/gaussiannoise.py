import cv2
import numpy as np
from numpy import shape
import random

def gaussian_noise(src,mean,sigma,percetage):
    noiseimg = src
    noisenum = int(src.shape[0]*src.shape[1]*percetage)
    for i in range(noisenum):
        randx = random.randint(0,src.shape[1]-1)
        randy = random.randint(0,src.shape[0]-1)
        noiseimg[randx,randy] = noiseimg[randx,randy]+random.gauss(mean,sigma)
        if noiseimg[randx,randy] < 0:
            noiseimg[randx, randy] = 0
        elif noiseimg[randx,randy] >255:
            noiseimg[randx, randy] = 255
    return noiseimg


img = cv2.imread("lenna.png",0)
img1 = gaussian_noise(img,2,4,0.01)
img3 = gaussian_noise(img,2,4,1)
cv2.imshow("gaosi",np.hstack([img1,img3]))
img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("sorce",np.hstack([img2,img3]))
cv2.waitKey()