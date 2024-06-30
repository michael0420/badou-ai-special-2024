import random

import cv2


def salt_and_pepper_noise(img, percetage):
    salt_pepper_noise_pic = img
    y, x = salt_pepper_noise_pic.shape[0], salt_pepper_noise_pic.shape[1]
    total_noise = int(percetage * y * x)
    for i in range(total_noise):
        random_x = random.randint(0, x - 1)
        random_y = random.randint(0, y - 1)

        random_v = random.random()
        print(random_v)

        if random_v > 0.5:
            random_v = 255
        else:
            random_v = 0

        salt_pepper_noise_pic[random_y][random_x] = random_v

    return salt_pepper_noise_pic

# flags=0 提出出来的是单通道的灰度图
img = cv2.imread("../../Lenna.jpg", flags=0)
cv2.imshow("img", img)

res = salt_and_pepper_noise(img, 0.2)
cv2.imshow("res", res)
cv2.waitKey()
