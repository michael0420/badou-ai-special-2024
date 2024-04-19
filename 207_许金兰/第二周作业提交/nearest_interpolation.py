"""
@author: 207-xujinlan
最近邻插值法缩放图片
"""

import cv2
import numpy as np


def nearest_interp(img, new_height, new_width):
    """
    图片缩放处理
    :param img: 原图
    :param new_height: 新图的高度
    :param new_width: 新图的宽度
    :return: 返回新图
    """
    height, width, chanels = img.shape  # 获取原图片的长宽和通道数
    new_img = np.zeros([new_height, new_width, chanels], np.uint8)
    sh = float(new_height) / height
    sw = float(new_width) / width
    for i in range(new_height):
        for j in range(new_width):
            x = int(i / sh + 0.5)  # 找原图中横轴最近坐标，取整
            y = int(j / sw + 0.5)  # 找原图中纵轴最近坐标，取整
            new_img[i, j] = img[x, y]  # 将原图最近坐标值赋值给新图
    return new_img


img_path = 'lenna.png'
img = cv2.imread(img_path)  # 读入原图
new_img = nearest_interp(img, 800, 800)  # 调用缩放函数
cv2.imshow("original image",img)  # 显示原图
cv2.imshow("new image",new_img)  # 显示缩放后的图
cv2.waitKey(0)
