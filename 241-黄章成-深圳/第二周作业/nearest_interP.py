import cv2
import numpy


# 最临近插值算法
def zoom_image(image, tar_size):
    h, w, c = image.shape
    temp_image = numpy.zeros((tar_size, tar_size, c), numpy.uint8)
    sh = tar_size / h
    sw = tar_size / w
    for i in range(tar_size):
        for j in range(tar_size):
            # x = int(i / sh + 0.5)  # 这里其实是四舍五入，和下面直接用round函数一样
            # y = int(j / sw + 0.5)
            x = int(round(i / sh))
            y = int(round(j / sw))
            temp_image[i, j] = image[x, y]
    return temp_image


img = cv2.imread('lenna.png')
tar_image = zoom_image(img, 1000)
# print(tar_image)
# print(tar_image.shape)

cv2.imshow('image', img)
cv2.imshow('tar_image', tar_image)
cv2.waitKey(0)
