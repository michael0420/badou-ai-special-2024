import random
import cv2

def PandS_niose(img,proportion):
    noise_img = img
    # 根据比例计算模糊像素点个数
    noise_num = int(proportion * noise_img.shape[0] * noise_img.shape[1])

    for i in range(noise_num):
    # 从图像上创建随机点，并且边缘不模糊
        randomX = random.randint(0,noise_img.shape[0]-1)
        randomY = random.randint(0,noise_img.shape[1]-1)

        if noise_img[randomX, randomY] < 122:
            noise_img[randomX, randomY] = 0
        else:
            noise_img[randomX, randomY] = 255

    return noise_img

img = cv2.imread("c4b591762800e7b417922ee4bcfb4cd.jpg",0)
cv2.imshow("img", img)
gauss_show = PandS_niose(img,0.5)
cv2.imshow("noise", gauss_show)
cv2.waitKey(0)
