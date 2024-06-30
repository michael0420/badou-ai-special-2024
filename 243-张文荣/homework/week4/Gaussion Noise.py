import cv2
import random

def _gaussian_noise(src,segma,means,percentage):
    noise_image = src
    noise_num = int(percentage * src.shape[0] * src.shape[1])
    for i in range(noise_num):
        randomX = random.randint(0, src.shape[0] - 1)
        randomY = random.randint(0, src.shape[1] - 1)
        noise_image[randomX,randomY] = noise_image[randomX,randomY] + random.gauss(means,segma)
        if noise_image[randomX,randomY] > 255:
            noise_image[randomX,randomY] = 255
        if noise_image[randomX,randomY] < 0:
            noise_image[randomX,randomY] = 0
    return noise_image


imag = cv2.imread('../week5/lenna.png', 0)
imag1 = _gaussian_noise(imag,4,2,0.8)

imag2 = cv2.imread('../week5/lenna.png', 0)
cv2.imshow('src',imag2)
cv2.imshow('gauss',imag1)
cv2.waitKey(0)