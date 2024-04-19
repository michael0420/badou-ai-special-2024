# _*_ coding: UTF-8 _*_
# @Time: 2024/4/3 19:51
# @Author: iris
import numpy as np
import math
import cv2


def bilinear_interpolation(src_image, dst_w, dst_h):
    """
    双线性插值
    :param src_image: 原图
    :param dst_w: 目标图行 x
    :param dst_h: 目标图列 y
    :return: 目标图
    """
    src_h, src_w, channels = src_image.shape
    dst_image = np.zeros([dst_h, dst_w, channels], np.uint8)
    # 计算缩放因子
    scale_x, scale_y = float(src_w / dst_w), float(src_h / dst_h)
    for channel in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # SrcX + 0.5 = (dstX + 0.5) * (srcWidth / dstWidth)
                # SrcY + 0.5 = (dstY + 0.5) * (srcHeight / dstHeight)
                # 根据中心几何对称计算，找到x，y在原图中的位置
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 计算x2,x1, y2,y1 向下取值
                src_x1 = int(math.floor(src_x))
                # min 做防呆检查 ceil向上取值
                src_x2 = min(int(math.ceil(src_x)), src_w - 1)
                # 计算x2,x1, y2,y1
                src_y1 = int(math.floor(src_y))
                # min 做防呆检查
                src_y2 = min(int(math.ceil(src_y)), src_h - 1)

                # 计算fQ1 和 fQ2 一斤fxy
                # f_R1 = (x2 - x) * f(Q11) + (x - x1) * f(Q21)
                f_R1 = (src_x2 - src_x) * src_image[src_y1, src_x1, channel] + (src_x - src_x1) * src_image[src_y1, src_x2, channel]
                # f_R2 = (x2 - x) * f(Q12) + (x - x1) * f(Q22)
                f_R2 = (src_x2 - src_x) * src_image[src_y2, src_x1, channel] + (src_x - src_x1) * src_image[src_y2, src_x2, channel]
                # 最终点 fxy = (y2 - y) * f_R1 + (y - y1) * f_R2
                fxy = (src_y2 - src_y) * f_R1 + (src_y - src_y1) * f_R2
                dst_image[dst_y, dst_x, channel] = fxy

    return dst_image


if __name__ == '__main__':
    image = cv2.imread('../data/lenna.png')
    # 进行双线性插值
    dst = bilinear_interpolation(image, 800, 800)
    cv2.imshow('image', image)
    cv2.imshow('bilinear_interpolation', dst)
    cv2.waitKey(0)
