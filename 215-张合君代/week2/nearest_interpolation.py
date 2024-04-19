# -*- coding: utf-8 -*-
"""
@author: zhjd

"""
import cv2
import numpy as np


def nearest_zoom_to_shape(image, new_shape):
    new_height, new_width = int(new_shape[0]), int(new_shape[1])
    height, width, channels = image.shape
    height_factor, width_factor = new_height / height, new_width / width
    new_height, new_width = int(height * height_factor), int(width * width_factor)
    new_image = np.zeros((new_height, new_width, channels), np.uint8)

    # 使用数组切片进行像素遍历
    for i in range(new_height):
        for j in range(new_width):
            new_image[i, j] = image[int(i / height_factor), int(j / width_factor)]

    return new_image


if __name__ == "__main__":
    before = cv2.imread('alex.jpg')
    nearest_zoom_to_shape = nearest_zoom_to_shape(before, (900, 900))
    # 保存缩放前后的图片
    cv2.imwrite('result/nearest_zoom_to_shape.jpg', nearest_zoom_to_shape)
