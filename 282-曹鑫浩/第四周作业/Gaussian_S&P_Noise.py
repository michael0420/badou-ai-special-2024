import cv2
import random

class Noise:
    def __init__(self, src, percentage):
        self.src = src
        self.percentage = percentage
        self._num = int(self.percentage * self.src.shape[0] * self.src.shape[1])
        print(self.src, self.percentage, self._num)

    def gaussian_noise(self, **kwargs):
        src = self.src
        for i in range(self._num):
            x = random.randint(0, src.shape[0]-1)
            y = random.randint(0, src.shape[1]-1)
            src[x, y] = src[x, y] + int(random.gauss(kwargs['mu'], kwargs['sigma']))
            if src[x, y] <= 0:
                src[x, y] = 0
            elif src[x, y] > 255:
                src[x, y] = 255
        print(src)
        return src

    def salt_pepper_noise(self):
        src = self.src
        for j in range(self._num):
            x = random.randint(0, src.shape[0]-1)
            y = random.randint(0, src.shape[1]-1)
            if random.random() <= 0.5:
                src[x, y] = 0
            elif random.random() > 0.5:
                src[x, y] = 255
        print(src)
        return src

img_src = cv2.imread('lenna.png', 0)
cv2.imshow('img_src', img_src)
gaussian_noise = Noise(img_src, 0.8)
img_gaussian_noise = gaussian_noise.gaussian_noise(mu=10, sigma=10)
cv2.imshow('gaussian noise', img_gaussian_noise)
salt_pepper_noise = Noise(img_src, 0.8)
img_salt_pepper_noise = salt_pepper_noise.salt_pepper_noise()
cv2.imshow('salt_pepper noise', img_salt_pepper_noise)
cv2.waitKey()
cv2.destroyAllWindows()
