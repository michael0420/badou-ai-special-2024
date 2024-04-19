#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2

#双线插值法（放大缩小图像）
def bilinear(src_img, dst_shape):
    # 计算目标图片到原图片的缩放比，且是图片坐标系的缩放，不是像素坐标系的缩放，像素位于图像像素格的中心
    src_h, src_w = src_img.shape[0], src_img.shape[1]
    dst_h, dst_w = dst_shape
    scale_h, scale_w = src_h / dst_h, src_w / dst_w  # 如果是像素坐标系的缩放则应该为 (src_h-1)/(dst_h-1)

    # 定义目标图片并向其中填充像素值，遍历目标图片像中的每个像素点
    dst_img = np.zeros((dst_h, dst_w, 3), np.uint8)
    for i in range(dst_h):
        for j in range(dst_w):
            # 将 目标像素坐标系下的坐标 --> 目标图像坐标系下的坐标(+0.5) --> 源图像坐标系下的坐标(*scale) --> 源像素坐标系下的坐标(-0.5)
            src_x = (j + 0.5) * scale_w - 0.5
            src_y = (i + 0.5) * scale_h - 0.5

            # 在非边界情况下获取左下角图像像素点坐标，在左/下边界的情况下保证大于等于0，在右/上边界的情况下保证小于等于src-2，以保证计算时所用的右上角像素坐标小于等于src-1
            src_x_int = min(max(int(src_x), 0), src_w - 2)
            src_y_int = min(max(int(src_y), 0), src_h - 2)

            # 获取所求像素点相比左下角像素点的距离
            src_x_float = src_x - src_x_int
            src_y_float = src_y - src_y_int

            # 计算每个像素值
            dst_img[i, j, :] = (1. - src_y_float) * (1. - src_x_float) * src_img[src_y_int, src_x_int, :] + \
                               (1. - src_y_float) * src_x_float * src_img[src_y_int, src_x_int + 1, :] + \
                               src_y_float * (1. - src_x_float) * src_img[src_y_int + 1, src_x_int, :] + \
                               src_y_float * src_x_float * src_img[src_y_int + 1, src_x_int + 1, :]

    return dst_img


if __name__ == "__main__":
    img_path = "lenna.png"

    src_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    dst_shape = (700, 700)

    # opencv的放缩图片函数
    resize_image = cv2.resize(src_img, dst_shape, interpolation=cv2.INTER_LINEAR)
    # 自定义的图片放缩函数
    dst_img = bilinear(src_img, dst_shape)

    cv2.imshow("src_img", src_img)
    # cv2.imshow("resize_image", resize_image)
    cv2.imshow("dst_img", dst_img)
    # 保存处理后的图片
    # cv2.imwrite("new_resize.jpg", dst_img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()  # Close all windows