"""
@author: jinlong

彩色图像放大 采用最临近插值算法（新图的每一个像素取用在原图最邻近的像素）
"""
import cv2
import numpy as np


def function(img):
    #switch =  10# True False
    height, width, channels = img.shape
    sh = 2 # 高度放大倍数
    sw = 2 # 宽度放大倍数
    aim_height = height * sh
    aim_width = width * sw
    emptyImage = np.zeros((aim_height, aim_width, channels), np.uint8)
    for pixel_height_index in range(aim_height):
        for pixel_width_index in range(aim_width):
            x = int(pixel_height_index / sh + 0.5)  # int(),转为整型，使用向下取整。
            y = int(pixel_width_index / sw + 0.5)

            if (x == height):
                x = height - 1
            if (y == width):
                y = width - 1

            #如果开启这个if,就仅仅把原图剪切放大，像素间无填充
            if 0:
                if (((pixel_height_index / sh) == x) and ((pixel_width_index / sw) == y)):
                    emptyImage[pixel_height_index, pixel_width_index] = img[x, y]
                else:
                    emptyImage[pixel_height_index, pixel_width_index] = 255
            else:
                emptyImage[pixel_height_index, pixel_width_index] = img[x, y]

    return emptyImage


# cv2.resize(img, (800,800,c),near/bin)

img = cv2.imread("lenna.png")
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)


