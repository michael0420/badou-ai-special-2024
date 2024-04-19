# 引入需要的库
# 灰度图公式 使用整数算法
# gray = (r*30+g*59+*b*11)/100;
# binary = gray/255 => [0,1]

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 获取图片看，返回一个矩阵
img = cv2.imread('./images/lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# rgb转为灰度图
def rgb2Gray(img):
    height, width = img.shape[:2]
    empty_img = np.zeros([height, width], img.dtype)

    # 循环图片的宽高，获取每个像素点信息
    for i in range(height):
        for j in range(width):
            #         当前物理点的信息,包括rgb信息[r,g,b]
            dot = img[i, j]
            # 根据公式转为灰度图，只有一个通道，只有黑白灰三种颜色，但是有0-255，256种值
            empty_img[i, j] = int((dot[0] * 30 + dot[1] * 59 + dot[2] * 11) / 100)
    return empty_img


# 二值图-把灰度图转为二值图
def gray2Binary(gray_img):
    rows, cols = gray_img.shape  # 只有一个通道，所有可以直接赋值
    empty_img = np.zeros([rows, cols], gray_img.dtype)

    for i in range(rows):
        for j in range(cols):
            # 对每个点进行二值化,每个点只有两个值，没有rgb通道
            if gray_img[i, j] / 255 <= 0.5:
                empty_img[i, j] = 0
            else:
                empty_img[i, j] = 1
    return empty_img


# 使用matplotlib，带有坐标信息以及其他功能
# sizeAndPos创建的尺寸和当前位置
def renderPic(sizeAndPos, img):
    plt.subplot(sizeAndPos)
    if (sizeAndPos == 222 or sizeAndPos == 223):
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)


# 分别获取灰度图，二值图
img_gray = rgb2Gray(img)
img_binary = gray2Binary(img_gray)
imgList = [img, img_gray, img_binary]

# 循环渲染
for i in range(len(imgList)):
    imgItem = imgList[i]
    sizeAndPos = int('22%d' % (i + 1))
    renderPic(sizeAndPos, imgItem)

plt.show()
