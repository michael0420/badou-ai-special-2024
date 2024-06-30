import cv2
import random


def pepper_salt_noise(img, persentage=0.3):
    img_copy = img.copy()
    height, width = img_copy.shape[:2]
    number = int(persentage * height * width)
    for i in range(number):
        x = random.randint(1, width - 1)
        y = random.randint(1, height - 1)
        if random.random() < 0.5:
            img_copy[y, x] = 0
        else:
            img_copy[y, x] = 255
    return img_copy


img_gray = cv2.imread('img/lenna.png', 2)
img_pepper_salt = pepper_salt_noise(img_gray, 0.3)
cv2.imshow("Gray Image", img_gray)
cv2.imshow('Pepper Salt Noise', img_pepper_salt)
cv2.waitKey(0)
