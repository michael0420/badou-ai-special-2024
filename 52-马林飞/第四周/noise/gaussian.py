import cv2
import random


def guassian_noise(img, ratio, sigma, means):
    h, w = img.shape[:2]
    noise_num = h * w * ratio
    for i in range(noise_num.__int__()):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        img[x][y] = img[x][y] + random.gauss(sigma, means)
        if img[x][y] > 255:
            img[x][y] = 255
        elif img[x][y] < 0:
            img[x][y] = 0
    return img


if __name__ == '__main__':
    src_img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('src', src_img)
    cv2.waitKey(3000)
    dest_img = guassian_noise(src_img, 0.8, 2, 4)
    cv2.imshow('dest', dest_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
