import random

import cv2


def gaussian_noise(img, mean, sigma, percetage):
    gaussian_pic = img
    y, x = img.shape[0], img.shape[1]
    noise_number = int(percetage * img.shape[0] * img.shape[1])
    for i in range(noise_number):
        # 寻找随机位置
        random_x = random.randint(0, x - 1)
        random_y = random.randint(0, y - 1)

        random_v = random.gauss(mean, sigma)
        gaussian_pic[random_y, random_x] = gaussian_pic[random_y, random_x] + random_v

        if gaussian_pic[random_y, random_x] > 255:
            gaussian_pic[random_y, random_x] = 255

        if gaussian_pic[random_y, random_x] < 0:
            gaussian_pic[random_y, random_x] = 0

    return gaussian_pic


img = cv2.imread("../../Lenna.jpg", 0)
cv2.imshow("source", img)

gaussian_pic = gaussian_noise(img, 9, 4, 0.8)
cv2.imshow("res", gaussian_pic)
cv2.waitKey()
