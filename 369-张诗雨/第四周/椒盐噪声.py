import cv2
import random


def salt_and_pepper_noise(img, percentage):
    noise_img = img     # shallow copy
    noise_num = int(percentage * img.shape[0] * img.shape[1])
    for i in range(noise_num):
        rand_x = random.randint(0, img.shape[0] - 1)
        rand_y = random.randint(0, img.shape[1] - 1)
        noise_img[rand_x, rand_y] = int(random.random() + 0.5) * 255    # 噪声为0或255
    return noise_img


if __name__ == '__main__':
    image = cv2.imread('lenna.png', 0)  # flag = 0, 灰度图
    cv2.imshow('source', image)
    noise_image = salt_and_pepper_noise(image, 0.8)
    cv2.imshow('salt and pepper noise', noise_image)
    cv2.waitKey(0)