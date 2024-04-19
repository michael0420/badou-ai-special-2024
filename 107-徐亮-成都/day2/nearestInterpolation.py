# 本质是算出新图像的坐标点和原图像坐标点的对应关系
import cv2
import numpy as np

# 获取图像
img = cv2.imread('./images/lenna.png')


def getImg(img):
    # 获取宽高和通道数
    width, height, channels = img.shape
    emptyImg = np.zeros((800, 800, channels), np.uint8)
    scale_w = 800 / width
    scale_y = 800 / height
    for i in range(800):
        for j in range(800):
            # 当前图像坐标对应原图像的坐标
            x = round(i / scale_w)
            y = round(j / scale_y)
            emptyImg[i, j] = img[x, y]
    return emptyImg

# 获取缩放后的图片
scaledImg = getImg(img)
# 把原图片和缩放后的图片一起展示出来
cv2.imshow('512x512', img)
cv2.imshow('800x800', scaledImg)
cv2.waitKey(0)
