import random

import cv2


def gaussian_noise(src, mu, sigma, percentage):
    w, h = src.shape[:2]
    new_img = src
    noise_num = int(w * h * percentage)
    for i in range(noise_num):
        rand_x = random.randint(1, w - 2)
        rand_y = random.randint(1, h - 2)
        new_pixel = src[rand_x, rand_y] + random.gauss(mu, sigma)
        new_pixel = min(255, max(0, new_pixel))
        new_img[rand_x, rand_y] = new_pixel
    return new_img


if __name__ == '__main__':
    img = cv2.imread('../lenna.png', 0)
    cv2.imshow('lenna', img)
    img_noised = gaussian_noise(img, 1, 3, 1)
    cv2.imshow('lenna_gaussian_noise', img_noised)
    cv2.waitKey(0)
