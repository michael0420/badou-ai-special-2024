import random

import cv2
def SaltPepperNoise(img, snr):
    s = int(img.shape[0] * img.shape[1] * snr)
    for i in range(s):
        x = random.randint(0, img.shape[0] - 1)
        y = random.randint(0, img.shape[1] - 1)

        if random.random() > 0.5:
            img[x,y] = 0
        else:
            img[x,y] = 255

    return img


if __name__ == '__main__':
    img = cv2.imread("lenna.png", 0)
    cv2.imshow("原图", img)
    salt_pepper_img = SaltPepperNoise(img,0.5)
    cv2.imshow("加椒盐噪声的图像",salt_pepper_img)
    cv2.waitKey()

