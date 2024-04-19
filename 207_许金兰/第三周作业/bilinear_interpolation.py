"""
@author: 207-xujinlan
双线性插值
"""

import numpy as np
import cv2

def bilinear_interpolation(img, dst_size):
    """
    双线性插值处理图片
    :param img: 原图
    :param dst_size: 目标图片，二位元组
    :return: dst_img,返回新图
    """
    src_width, src_height, channels = img.shape
    # 如果新图大小和原图一样，直接复制原图，不用重新计算
    if src_width == dst_size[0] and src_height == dst_size[1]:
        dst_img = img.copy()
    else:
        dst_img = np.zeros((dst_size[0], dst_size[1], channels), dtype='uint8')   #设置空白新图
    scale_x = float(src_width) / dst_size[0]
    scale_y = float(src_height) / dst_size[1]
    for c in range(channels):
        for i in range(dst_size[0]):
            x = (i + 0.5) * scale_x - 0.5
            x0 = int(np.floor(x))
            x1 = min(src_width - 1, x0 + 1)
            for j in range(dst_size[1]):
                y = (j + 0.5) * scale_y - 0.5
                y0 = int(np.floor(y))
                y1 = min(src_height - 1, y0 + 1)
                f1 = (x1 - x) * img[x0, y0, c] + (x - x0) * img[x1, y0, c]
                f2 = (x1 - x) * img[x0, y1, c] + (x - x0) * img[x1, y1, c]
                dst_img[i, j, c] = int((y1 - y) * f1 + (y - y0) * f2)
    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst_size = (800, 800)
    dst_img = bilinear_interpolation(img, dst_size)
    cv2.imshow('bilinear_interpolation', dst_img)
    cv2.waitKey(0)
