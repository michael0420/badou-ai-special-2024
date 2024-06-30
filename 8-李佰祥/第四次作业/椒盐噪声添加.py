import random

import cv2
import numpy
#指定信噪比
img = cv2.imread("../../lenna.png" , 0)
percetage = 0.05

noiseNum = int(percetage * img.shape[0]*img.shape[1])
print(noiseNum)

for i in range(noiseNum):
    randX = random.randint(0,img.shape[0] - 1)
    randY = random.randint(0,img.shape[1] - 1)

    if random.random() <=0.5:
        img[randX,randY] = 0
    else :
        img[randX, randY] = 255



cv2.imshow("img",img)
cv2.waitKey(0)










