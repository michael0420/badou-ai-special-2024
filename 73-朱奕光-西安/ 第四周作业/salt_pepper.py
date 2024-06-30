import cv2
import matplotlib.pyplot as plt
import random


def salt_pepper(scr, percet):
    img_salt_pepper = scr
    img_salt_pepper_piexl = int(img_salt_pepper.shape[0] * img_salt_pepper.shape[1] * percet)
    for i in range(img_salt_pepper_piexl):
        randomX = random.randint(0, img_salt_pepper.shape[0] - 1)
        randomY = random.randint(0, img_salt_pepper.shape[1] - 1)
        randomfloat = random.random()
        if randomfloat <= 0.5:
            img_salt_pepper[randomX, randomY] = 0
        else:
            img_salt_pepper[randomX, randomY] = 255
    return img_salt_pepper


img = cv2.imread('lenna.png', 0)
plt.subplot(121)
plt.imshow(img, cmap='gray')
img_result = salt_pepper(img, 0.2)
plt.subplot(122)
plt.imshow(img_result, cmap='gray')
plt.show()
