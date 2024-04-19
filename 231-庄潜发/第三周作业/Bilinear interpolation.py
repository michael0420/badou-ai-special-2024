"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/4/7 22:27
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def Bilinear_interpolation():
    # 获取原图的shape-->高度 宽度 通道
    src_height, src_width, src_channels = img.shape
    # 生成空白图片, 800*800
    new_img = np.zeros((800, 800, src_channels), dtype=np.uint8)
    # 生成比例: 原/目标
    height_sh = float(src_height / 800)
    width_sh = float(src_width / 800)
    # 遍历坐标
    for channel in range(src_channels):
        for dist_height in range(800):
            for dist_width in range(800):
                # 整体居中,获取坐标
                x = (dist_width + 0.5) * width_sh - 0.5
                y = (dist_height + 0.5) * height_sh - 0.5
                # 获取x0, x1, y0, y1
                x0 = int(np.floor(x))
                x1 = min(x0 + 1, src_width - 1)
                y0 = int(np.floor(y))
                y1 = min(y0 + 1, src_height - 1)
                # 计算新坐标像素值
                new_img[dist_height, dist_width, channel] = int(
                    (y1 - y) * (x1 - x) * img[y0, x0, channel] + (y1 - y) * (x - x0) * img[y0, x1, channel] +
                    (y - y0) * (x1 - x) * img[y1, x0, channel] + (y - y0) * (x - x0) * img[y1, x1, channel])
    return new_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    new_img = Bilinear_interpolation()
    cv2.imshow("img", img)
    cv2.imshow("new_img", new_img)
    cv2.waitKey()
