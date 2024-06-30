import cv2
import random

img = cv2.imread("lenna.png", 0)
h = img.shape[0]
w = img.shape[1]
# 获取要高斯噪声的像素点数量
noise_count = int(0.1 * h * w)
for i in range(noise_count):
    # 获取随机像素点位置
    rand_x = random.randint(0, h - 1)
    rand_y = random.randint(0, w - 1)

    if random.random() <= 0.5:
        img[rand_x, rand_y] = 0
    else:
        img[rand_x, rand_y] = 255

cv2.imshow("lenna", img)
cv2.waitKey(0)

