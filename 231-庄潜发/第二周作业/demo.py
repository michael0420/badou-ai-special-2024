"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/3/27 20:51
"""
import cv2
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt


class IG():

    def __init__(self):
        # opencv读取的图片通道排序为BGR
        self.img = cv2.imread("lenna.png")

    def img_gray_way1(self):
        # 获取图片的高和宽, 第三个参数是通道,shape:矩阵的意思
        h, w = self.img.shape[:2]
        logger.info(f"-------------输出图片矩阵的宽和高:{h},{w}")
        # 创建一张和当前图片大小一样的单通道图片,
        # zeros函数表示创建矩阵, 一个参数传入想要创建矩阵shape的大小(图像的矩阵),dtype表示创建新矩阵的数据类型
        img_gray = np.zeros([h, w], self.img.dtype)
        for i in range(h):
            for j in range(w):
                # 取出当前图片h和w中的BGR坐标, 计算当前像素(h,w)的BGR的三个[125 137 226]坐标, 将这个坐标通过浮点计算转成灰色图坐标,存储进新图
                m = self.img[i, j]
                # 将BGR坐标转换成gray坐标并赋值给新图像
                img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
        logger.info(f"-------------计算实现的灰度图矩阵:{img_gray}")
        # # 展示图片
        # cv2.imshow("img show 1", img_gray)
        # # 解决图片闪退问题
        # cv2.waitKey()
        plt.subplot(221)  # 2x2 个位置的第一个图片
        plt.imshow(img_gray, cmap='gray')

    def img_gray_way2(self):
        # 通过调用方法实现灰度图
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        logger.info(f"-------------调用方法实现的灰度图矩阵:{img_gray}")
        # # 展示图片
        # cv2.imshow("img show 2", img_gray)
        # # 解决图片闪退问题
        # cv2.waitKey()
        plt.subplot(222)
        plt.imshow(img_gray, cmap='gray')

    def img_binary_way1(self):
        # 二值化需要先将色彩图片转换成灰度图,再进行二值转换
        # 将图片转换成灰度图
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # threshold方法, 第一个参数(灰度图图片), 127(阈值对比,越大对黑色判定条件越宽松), 255(默认赋值),cv2.THRESH_BINARY(阈值类型)
        # 返回两个值, 一个是当前对比值, 一个是当前图片的矩阵
        ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        logger.info(f"-------------二值图当前对比度:{ret}")
        logger.info(f"-------------二值图矩阵:{img_binary}")
        # # 展示图片
        # cv2.imshow("img show 3", img_binary)
        # # 解决图片闪退问题
        # cv2.waitKey()
        plt.subplot(223)
        plt.imshow(img_binary, cmap='gray')

    def img_binary_way2(self):
        # 使用numpy实现
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img_binary = np.where(img_gray >= 128, 255, 0).astype(np.uint8)
        # img_binary = np.where(img_gray >= 128, 1, 0)
        logger.info(f"-------------二值图矩阵:{img_binary}")
        # 展示图片,astype(np.uint8)将数据类型转换成无符号的8位整型,这种类型的整数范围是0-255,用于图像处理
        # img_binary = (img_binary * 255).astype(np.uint8)
        # cv2.imshow("img show 4", img_binary)
        # # 解决图片闪退问题
        # cv2.waitKey()
        plt.subplot(224)
        plt.imshow(img_binary, cmap='gray')


if __name__ == '__main__':
    ig = IG()
    ig.img_gray_way1()
    ig.img_gray_way2()
    ig.img_binary_way1()
    ig.img_binary_way2()
    plt.show()
