import cv2
import random
import copy
import numpy as np

def gaussian_noise(image, m, s, per):
    gaussian = copy.deepcopy(image)
    height = gaussian.shape[0]
    width = gaussian.shape[1]
    num = int(height * width * per)
    for i in range(num):
        h = random.randint(0, height - 1)
        w = random.randint(0, width - 1)
        value = gaussian[h][w] + random.gauss(m, s)
        value = min(value, 255)
        value = max(0, value)
        gaussian[h][w] = value
    return gaussian


def pepper_noise(image, per):
    pepper = copy.deepcopy(image)
    height = pepper.shape[0]
    width = pepper.shape[1]
    num = int(height * width * per)
    for i in range(num):
        h = random.randint(0, height - 1)
        w = random.randint(0, width - 1)
        value = random.choice([0, 255])
        pepper[h][w] = value
    return pepper


img = cv2.imread("../lenna.png", 0)

gaussian_img = gaussian_noise(img, 2, 10, 0.9)
pepper_img = pepper_noise(img, 0.5)

result = np.hstack((img, gaussian_img, pepper_img))
cv2.imshow("result", result)
cv2.imwrite('result_img.png',result)
cv2.waitKey(0)
