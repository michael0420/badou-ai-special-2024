"""
使用双线性插值法，改变图片的尺寸


"""

import cv2
import numpy as np

src_img = cv2.imread("../lenna.png")
cv2.imshow("src_img", src_img)
src_img_x, src_img_y, channel = src_img.shape
tar_img_x, tar_img_y = 800, 800
if tar_img_x == src_img_x and tar_img_y == src_img_y:
    print("尺寸相同，无需改变")
else:
    print("尺寸不同，使用双线性插值法改变尺寸")
    # 创建一张和目标尺寸一样的图片
    tar_img = np.zeros((tar_img_x, tar_img_y, channel), dtype=np.uint8)
    # 计算改变图片的比例
    x_ratio, y_ratio = src_img_x / tar_img_x, src_img_y / tar_img_y
    # 遍历目标图片的每一个像素点
    for i in range(channel):
        for y in range(tar_img_y):
            for x in range(tar_img_x):
                # 计算目标图中的每一个像素点对应的原图中的坐标
                x_src = x_ratio * (x + 0.5) - 0.5
                y_src = y_ratio * (y + 0.5) - 0.5

                # 严谨一点，判断新坐标后的相邻的4个点是否超出原图的边界，如果超出就取原图的长宽
                x_src_ceil_x0 = int(x_src)
                x_src_ceil_x2 = min(x_src_ceil_x0 + 1, src_img_x - 1)
                y_src_ceil_x0 = int(y_src)
                y_src_ceil_y2 = min(y_src_ceil_x0 + 1, src_img_y - 1)

                # 开始带入公式计算
                temp0 = (x_src_ceil_x2 - x_src) * src_img[y_src_ceil_x0, x_src_ceil_x0, i] + (x_src - x_src_ceil_x0) * \
                        src_img[y_src_ceil_y2, x_src_ceil_x0, i]
                temp1 = (x_src_ceil_x2 - x_src) * src_img[y_src_ceil_x0, x_src_ceil_x2, i] + (x_src - x_src_ceil_x0) * \
                        src_img[y_src_ceil_y2, x_src_ceil_x2, i]
                tar_img[y, x, i] = (y_src_ceil_y2 - y_src) * temp0 + (y_src - y_src_ceil_x0) * temp1

    cv2.imshow("tar_img", tar_img)
    cv2.waitKey(0)
