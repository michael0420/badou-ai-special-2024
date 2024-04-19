from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def imgProcess1(img):
    img_gray = rgb2gray(img)
    plt.subplot(211)
    plt.imshow(img_gray, cmap='gray')

    img_2 = np.where(img_gray > 0.5, 0, 1)
    plt.subplot(212)
    plt.imshow(img_2, cmap='gray')
    plt.show()


def imgProcess2(img):
    h, w, c = img.shape
    new_h = h * 2
    new_w = w * 2
    new_img = np.zeros((new_h, new_w, c), np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            x = min(h - 1, round(i / 2))
            y = min(w - 1, round(j / 2))
            new_img[i][j] = img[x][y]
    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.imshow(new_img)
    plt.show()


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    imgProcess1(img)
    imgProcess2(img)
