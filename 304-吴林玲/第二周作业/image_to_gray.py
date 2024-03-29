from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


#实现灰度化
def imag_to_gray():
    img = cv2.imread("lenna.png")
    h, w = img.shape[:2]
    gray_img = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            gray_img[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

    plt.subplot(221)
    plt.imshow(gray_img, cmap='gray')



#实现二值化
def imag_to_black():
    img = plt.imread("lenna.png")
    h, w = img.shape[:2]
    black_img = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            black_img[i,j] = m[0]*0.11 + m[1]*0.59 + m[2]*0.3

    rows, cols = black_img.shape
    for i in range(rows):
         for j in range(cols):
             if (black_img[i, j] <= 0.5):
                 black_img[i, j] = 0
             else:
                 black_img[i, j] = 1

    plt.subplot(222)
    plt.imshow(black_img, cmap='gray')





if __name__ == '__main__':
    imag_to_gray()
    imag_to_black()
    plt.show()
