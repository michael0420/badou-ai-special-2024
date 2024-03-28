import cv2
import numpy
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


def image2gray():
    # 读取图片
    img = cv2.imread("lenna.png")
    # 获取图片像素
    h, w = img.shape[:2]
    # 创建一张和原图一样大小的单通道图片
    new_img = numpy.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            # 遍历每个像素，将其转为单通道值，并赋值到新的单通道图上
            new_img[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
    print(new_img)
    # 输出图片
    cv2.imshow("this is new image", new_img)


image2gray()


def image2gray2():
    # 读取图片
    img = plt.imread("lenna.png")
    # 调用方法rgb转灰度
    gary_image = rgb2gray(img)
    # 返回灰度图
    return gary_image


# 灰度转二值
def gray2binary():
    # 调用方法获取灰度图
    gray_img = image2gray2()
    # 获取像素
    rows, cols = gray_img.shape
    # 创建一个新的单通道图，存放二值图
    binary_img = numpy.zeros([rows, cols], gray_img.dtype)
    # 遍历灰度图每个像素点，转为二值
    for i in range(rows):
        for j in range(cols):
            if gray_img[i, j] >= 0.5:
                binary_img[i, j] = 1
            else:
                binary_img[i, j] = 0
    plt.imshow(binary_img, cmap='binary')
    plt.show()


gray2binary()
