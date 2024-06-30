import cv2
import random
import matplotlib.pyplot as plt


def gauss(src, mu, sigma, percet):
    gaussimg = src
    gausspiexl = int(gaussimg.shape[0] * gaussimg.shape[1] * percet)
    for i in range(gausspiexl):
        gaussX = random.randint(0, gaussimg.shape[0] - 1)
        gaussY = random.randint(0, gaussimg.shape[1] - 1)
        gaussimg[gaussX, gaussY] = gaussimg[gaussX][gaussY] + random.gauss(mu, sigma)
    return gaussimg


img = cv2.imread('lenna.png', 0)
plt.subplot(121)
plt.imshow(img, cmap='gray')
img_result = gauss(img, 4, 8, 0.8)
plt.subplot(122)
plt.imshow(img_result, cmap='gray')
plt.show()