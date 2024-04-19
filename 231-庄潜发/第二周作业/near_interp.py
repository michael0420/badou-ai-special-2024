"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/3/29 23:18
"""
import cv2
import numpy as np
from loguru import logger


def near_interp():
    # 获取原图片的高,宽,通道
    old_height, old_width, channels = img.shape
    # 创建一个上采样后的图片,高800,宽800,通道与原图一样
    empty_img = np.zeros((800, 800, channels), np.uint8)
    # 计算宽和高的放大比例
    height_scale = 800/old_height
    width_scale = 800/old_width
    for i in range(800):  # 高
        for j in range(800):  # 宽
            # 新图像像素/比例得到最邻近的原图像的像素点,
            # 除法取得出的结果是浮点型,需要转为int型,但int型是向下取整,
            # 我们需要四舍五入,加0.5的话就相当于 ---- int(1.7+0.5) -> 2  int(1.2+0.5) -> 1
            # 也可以通过round()方法进行取最近的像素点
            x = int(i/height_scale + 0.5)
            y = int(j/width_scale + 0.5)
            # x = round(i/height_scale)
            # y = round(j/width_scale)
            # 求出i和j最邻近的x和y像素的位置,通过img[x,y]取到最邻近的像素值赋值给img[i,j]
            empty_img[i, j] = img[x, y]
    return empty_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    new_img = near_interp()
    logger.info(f"-------------原图像矩阵{img}")
    logger.info(f"-------------最邻近插值计算上取样后图像矩阵{new_img}")
    cv2.imshow("old_img", img)
    cv2.imshow("new_img", new_img)
    cv2.waitKey()
