import cv2
from skimage import util
import numpy as np
import random

def GaussNoise(img, mean, sigma, percetage):
    noise_img = np.array(img)
    high, width = img.shape[:2]
    noise_num = int(percetage * img.shape[0] * img.shape[1])
    for i in range(noise_num):
        randint_x = random.randint(1, high - 1)
        randint_y = random.randint(1, width - 1)
        noise_img[randint_x, randint_y] = noise_img[randint_x, randint_y] + random.gauss(mean, sigma)
        if noise_img[randint_x, randint_y] < 0:
            noise_img[randint_x, randint_y] = 0
        if noise_img[randint_x, randint_y] > 255:
            noise_img[randint_x, randint_y] = 255
    return noise_img

img_gray = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)
img_gauss_noise = GaussNoise(img_gray, 5, 0.2, 0.5)
cv2.imshow("img_gray", img_gray)
cv2.imshow("img_gauss_noise", img_gauss_noise)
cv2.waitKey(0)
