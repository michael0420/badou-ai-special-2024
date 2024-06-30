import random

import cv2 as cv
import numpy as np


def GaussianNoise(src, means, sigma, rate):
    noise_img = src
    noise_num = int(noise_img.shape[0] * noise_img.shape[1] * rate)
    for i in range(noise_num):
        noise_x = random.randint(0, src.shape[0] - 1)
        noise_y = random.randint(0, src.shape[1] - 1)
        noise_img[noise_y, noise_x] = noise_img[noise_y, noise_x] + random.gauss(means, sigma)
        if noise_img[noise_y, noise_x] > 255:
            noise_img[noise_y, noise_x] = 255
        elif noise_img[noise_y, noise_x] < 0:
            noise_img[noise_y, noise_x] = 0
    return noise_img


if __name__ == '__main__':
    gray = cv.imread('../lenna.png', 0)
    img = cv.imread('../lenna.png')
    gray1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    result = GaussianNoise(gray, 0.8, 2, 0.8)
    cv.imshow("result", np.hstack([result, gray1]))
    cv.waitKey(0)
    cv.destroyAllWindows()
