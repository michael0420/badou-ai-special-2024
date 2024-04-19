import cv2
import random

def GaussianNoise(src_img, mean, sigma, percetage):
    des_img = src_img.copy()
    noise_num  = int(src_img.shape[0] * src_img.shape[1] * percetage)
    for i in range(noise_num):
        src_x = random.randint(0, des_img.shape[0] - 1)
        src_y = random.randint(0, des_img.shape[1] - 1)

        des_img[src_x, src_y] = des_img[src_x, src_y] + random.gauss(mean, sigma)
        if des_img[src_x, src_y] < 0:
            des_img[src_x, src_y] = 0
        if des_img[src_x, src_y] > 255:
            des_img[src_x, src_y] = 255
    return des_img

src_img = cv2.imread('lenna.png', 0)
des_img = GaussianNoise(src_img, 2, 5, 3)
cv2.imshow("src img", src_img)
cv2.imshow("des img", des_img)
cv2.waitKey(0)