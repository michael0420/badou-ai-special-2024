import cv2
import numpy as np
import random


def gaussian_noise(img, mean, sigma, percentage):
    noise_img = img     # shallow copy
    noise_num = int(percentage * img.shape[0] * img.shape[1])
    for i in range(noise_num):
        rand_x = random.randint(0, img.shape[0] - 1)
        rand_y = random.randint(0, img.shape[1] - 1)
        noise_img[rand_x, rand_y] += random.gauss(mean, sigma)
    noise_img = np.clip(noise_img, 0, 255)      # 噪声在0到255之间
    return noise_img


if __name__ == '__main__':
    image = cv2.imread('lenna.png', 0)  # flag = 0, 灰度图
    noise_image = gaussian_noise(image, 2, 4, 0.8)
    cv2.imshow('gaussian noise', noise_image)
    cv2.waitKey(0)
