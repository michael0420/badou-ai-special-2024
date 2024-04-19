# -*- coding: utf-8 -*-
"""
 作业：2.实现最临近插值
"""
import cv2
import numpy as np


def image_zoom(target_h=800, target_w=800):
    """
    通过临近插值法实现图片放大、缩小
    :param target_h: 图片高度
    :param target_w: 图片宽带
    :return:
    """
    # 加载图片
    img = cv2.imread("lenna.png")
    # 获取图片原始高、宽、通道数
    h, w, channel = img.shape

    # 计算缩放比
    h_zoom_ratio = target_h / h
    w_zoom_ratio = target_w / w
    # 生成指定大小空数组，且数据类型是uint8
    new_img = np.zeros((target_h, target_w, channel), np.uint8)
    for i in range(target_h):
        for j in range(target_w):
            # 新值/缩放比-> 原始值，通过+0.5实现四舍五入
            x = int(i / h_zoom_ratio + 0.5)
            y = int(j / w_zoom_ratio + 0.5)
            # 将原图中的值赋值到新图的坐标中
            new_img[i, j] = img[x, y]
    cv2.imshow("new image", new_img)
    cv2.waitKeyEx(0)


if __name__ == '__main__':
    image_zoom(300, 300)
