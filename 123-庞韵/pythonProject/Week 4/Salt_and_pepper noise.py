# pepper noise = 0 (low gray density), salt noise = 255 (high gray density)
# 1. define a SIGNAL-NOISE RATION (SNR) which in the range btw [0,1]
# 2. get the total pixel number, then calculate the total pixel number that need to add noise
# 3. random pick pixel point
# 4. set pixel value to 0 or 255

import numpy as np
import cv2
from numpy import shape
import random


def salt_and_pepper(src, percentage):

    NoiseImage = src.copy()
    NoiseNum = int(percentage * src.shape[0] * src.shape[1]) # get the total pixel number

    for i in range(NoiseNum):
        # pick random pixel point
        randX = random.randint(0, src.shape[0]-1) # -1 to avoid edge
        randY = random.randint(0, src.shape[1]-1)
        # set the pixel value of the random point to 0 or 255
        # use random.random to randomly assign 0 or 255 to the pixel point
        if random.random() < 0.5:
            NoiseImage[randX, randY] = 0
        else:
            NoiseImage[randX, randY] = 255
    return NoiseImage


# load the image and apply the function
img = cv2.imread('../images/lenna.png', 0)
img1 = salt_and_pepper(img, 0.8)
cv2.imwrite('salt_and_pepper.png', img1)
cv2.imshow('salt_and_pepper', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()



