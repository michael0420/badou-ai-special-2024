import cv2
import skimage
import matplotlib.pyplot as plt
import random

# 椒盐噪声

def salt_noise(img_src, ratio):
    imgNoise = img_src
    noiseNum = int(ratio * img_src.shape[0] * img_src.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img_src.shape[0] - 1)
        randY = random.randint(0, img_src.shape[1] - 1)
        if random.random()<=0.5:
            imgNoise[randX, randY] = 0
        else:
            imgNoise[randX, randY] = 255
    return imgNoise

img_src = cv2.imread("../lenna.png", 0)
img_salt = salt_noise(img_src, 0.2)
img_salt_ski = skimage.util.random_noise(img_src, "s&p")

img_src = cv2.imread("../lenna.png")
img_src_1 = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
plt.subplot(221)
plt.imshow(img_src_1)
plt.title("src")

plt.subplot(222)
plt.imshow(img_salt)
plt.title("salt")

plt.subplot(223)
plt.imshow(img_salt_ski)
plt.title("salt_ski")

plt.show()
