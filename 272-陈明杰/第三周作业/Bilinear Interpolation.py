#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2


def bilinear_interpolation(src: np.ndarray, out_dim):
    # 目标图像的高度
    dst_high = out_dim[0]
    # 目标图像的宽度
    dst_width = out_dim[1]
    # 取出原图像的高度，宽度，通道数
    src_high, src_width, channels = src.shape
    # 计算高度和宽度被放大或者缩小的倍数，即比例
    ratio_high = dst_high / src_high
    ratio_width = dst_width / src_width
    # 如果目标图像的high和width都和原图像的high和width一样
    # 那么就直接返回原图的拷贝即可
    if dst_high == src_high and src_high == src_width:
        return src.copy()

    # 创建一个和原图一样大小的多通道矩阵
    emptyImg = np.zeros((dst_high, dst_width, channels), src.dtype)

    for channel in range(channels):
        for i in range(dst_high):
            for j in range(dst_width):
                # 计算新矩阵i,j下标在原矩阵中的对应位置，按照比例变回去就行
                # 这里涉及到一个数学公式，对下标修正过
                # src_x+0.5=(i+0.5)/ratio_high
                # src_y+0.5=(j+0.5)/ratio_width
                # 具体推导公式参考第二节课的课件
                src_x = float(i + 0.5) / ratio_high - 0.5
                src_y = float(j + 0.5) / ratio_width - 0.5

                # 根据目标矩阵计算出在原矩阵的对应位置之后，找到
                # 最接近这个点的四个整数点，因为计算出来的在原矩
                # 阵中的位置可能是小数，向下取整即可取到左边的点
                src_x1 = int(src_x)
                # 防止越界，需要做边界修正
                src_x2 = int(min(src_x1 + 1, src_width - 1))
                src_y1 = int(src_y)
                src_y2 = int(min(src_y1 + 1, src_high - 1))

                # 根据课件的公式，按照加权平均即可计算出i,j位置对应的像素值
                tmp1 = (src_x2 - src_x) * src[src_x1, src_y1, channel] + (src_x - src_x1) * src[src_x2, src_y1, channel]
                tmp2 = (src_x2 - src_x) * src[src_x1, src_y2, channel] + (src_x - src_x1) * src[src_x2, src_y2, channel]
                # 填写到新矩阵的对应的位置中
                emptyImg[i, j, channel] = (src_y2 - src_y) * tmp1 + (src_y - src_y1) * tmp2
    # 返回新矩阵
    return emptyImg


def main():
    src = cv2.imread("lenna.png")
    newHigh = int(input("请输入新图像的high："))
    newWidth = int(input("请输入新图像的width："))

    dst = bilinear_interpolation(src, (newHigh, newWidth))
    cv2.imshow("before", src)
    cv2.imshow("after", dst)
    cv2.waitKey(0)


main()
