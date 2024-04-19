import cv2
import numpy as np


def zoom_image(img_path, zoom_height, zoom_width):
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    empty_img = np.zeros((zoom_height, zoom_width, channels), np.uint8)
    sh = zoom_height / height
    sw = zoom_width / width
    for i in range(zoom_height):
        for j in range(zoom_width):
            empty_img[i, j] = img[int(i / sh + 0.5), int(j / sw + 0.5)]
    return empty_img
