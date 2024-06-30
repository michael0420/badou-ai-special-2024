import cv2
import numpy as np
import random


def SaltPepperNoise(img, percetage):
    noise_img = np.array(img)
    high, width = img.shape[:2]
    noise_num = int(percetage * img.shape[0] * img.shape[1])
    for i in range(noise_num):
        randint_x = random.randint(1, high - 1)
        randint_y = random.randint(1, width - 1)

        if random.random() > 0.5:
            noise_img[randint_x, randint_y] = 0
        else:
            noise_img[randint_x, randint_y] = 255
    return noise_img


img_gray = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("img_gray", img_gray)
img_noise = SaltPepperNoise(img_gray, 0.5)
cv2.imshow("img_noise", img_noise)

cv2.waitKey(0)
