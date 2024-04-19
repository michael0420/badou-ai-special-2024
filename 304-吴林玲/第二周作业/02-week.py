import cv2
import numpy as np
import matplotlib.pyplot as plt

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



#实现最临近插值
def function(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((800,800,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh + 0.5)  #int(),转为整型，使用向下取整。
            y=int(j/sw + 0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage


if __name__ == '__main__':
    imag_to_gray()
    imag_to_black()
    img = cv2.imread("lenna.png")
    zoom = function(img)
    cv2.imshow("nearest interp", zoom)
    plt.show()
    cv2.waitKey(0)
