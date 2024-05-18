import random
import cv2


def gaussn_iose(img, proportion, mu, sigma):
    noise_img = img
    # 根据比例计算高斯模糊像素点个数
    noise_num = int(proportion * noise_img.shape[0] * noise_img.shape[1])

    for i in range(noise_num):
    # 从图像上创建随机点，并且边缘不高斯模糊
        randomX = random.randint(0, noise_img.shape[0] - 1)
        randomY = random.randint(0, noise_img.shape[1] - 1)

        # 在原有像素值上加上随机数
        noise_img[randomX, randomY] += random.gauss(mu, sigma)

        if noise_img[randomX, randomY] > 255:
            noise_img[randomX, randomY] = 255

        elif noise_img[randomX, randomY] < 0:
            noise_img[randomX, randomY] = 0

        return noise_img


img = cv2.imread("c4b591762800e7b417922ee4bcfb4cd.jpg", 0)
cv2.imshow("img", img)
gauss_show = gaussn_iose(img, 0.9, 4, 4)
cv2.imshow("gauss", gauss_show)
cv2.waitKey(0)
