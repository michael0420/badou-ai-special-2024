import cv2
import skimage
import matplotlib.pyplot as plt
import random
from numpy import shape

img_src = cv2.imread("../lenna.png", 0)
# img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
'''一、skimage函数实现'''
# 高斯噪声
img_gaussian = skimage.util.random_noise(img_src, "gaussian", mean=0.1, var=0.2)

# 泊松噪声
img_poisson = skimage.util.random_noise(img_src, "poisson")

'''二、底层原理'''

def gaussian_noise(src, means, sigma, ratio):
    noiseImg = src
    noiseNum = int(ratio * src.shape[0] * src.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        noiseImg[randX, randY] = noiseImg[randX, randY] + random.gauss(means, sigma)
        if noiseImg[randX, randY] < 0:
            noiseImg[randX, randY] = 0
        elif noiseImg[randX, randY] > 255:
            noiseImg[randX, randY] = 255
    return noiseImg


img_gaussian_2 = gaussian_noise(img_src, 40, 80, 0.8)

plt.subplot(221)
plt.imshow(img_src)
plt.title("src")

plt.subplot(222)
plt.imshow(img_gaussian)
plt.title("gaussian")

plt.subplot(223)
plt.imshow(img_poisson)
plt.title("poisson")

plt.subplot(224)
plt.imshow(img_gaussian_2)
plt.title("gaussian_2")

plt.show()
