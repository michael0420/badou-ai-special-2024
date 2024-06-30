import random

import cv2


def pepper_salt_noise(src, percentage):
    w, h = src.shape[:2]
    new_img = src
    noise_num = int(w * h * percentage)
    for i in range(noise_num):
        rand_x = random.randint(1, w - 2)
        rand_y = random.randint(1, h - 2)
        new_img[rand_x, rand_y] = 255 if random.randint(0, 1) == 1 else 0
    return new_img


if __name__ == '__main__':
    img = cv2.imread('../lenna.png', 0)
    cv2.imshow('lenna', img)
    img_noised = pepper_salt_noise(img, 0.05)
    cv2.imshow('lenna_pepper_salt_noise', img_noised)
    cv2.waitKey(0)
