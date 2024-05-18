import cv2
import random


def GaussianNoise(img, mean, sigma, percentage):
    noise_img = img
    pixel_num = img.shape[0] * img.shape[1]
    noise_num = int(percentage * pixel_num)
    for i in range(noise_num):
        random_x = random.randint(0, img.shape[0] - 1)
        random_y = random.randint(0, img.shape[1] - 1)
        noise_img[random_x, random_y] = noise_img[random_x, random_y] + random.gauss(mean, sigma)
        if noise_img[random_x, random_y] < 0:
            noise_img[random_x, random_y] = 0
        elif noise_img[random_x, random_y] > 255:
            noise_img[random_x, random_y] = 255

    return noise_img


img = cv2.imread("./lenna.png", 0)  # 0灰度读取, 默认1彩色读取
noise_img = GaussianNoise(img, 1, 5, 0.9)
cv2.imshow("gaussian img", noise_img)

img = cv2.imread("./lenna.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("src img", gray_img)

cv2.waitKey(0)
