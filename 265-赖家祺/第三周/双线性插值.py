# 1. 实现双线性插值
# 2. 实现直方图均衡化
# 3. 实现sobel边缘检测

import datetime
import cv2
import numpy as np
from PIL import Image

"""
双线性插值，掌握几何中心对称
"""


def bilinear(img, new_size):
    dst_width, dst_height = new_size[0], new_size[1]

    # 原图像
    src_height, src_width, src_channel = img.shape
    print(f"source height={src_height}, width={src_width}, channel={src_channel}")

    # 目标图像规格的空矩阵, dtype可以=np.uint8
    dst_img = np.zeros((dst_height, dst_width, 3), dtype=img.dtype)

    # 缩放比例
    scale_y = src_height / dst_height
    scale_x = src_width / dst_width

    for i in range(src_channel):
        for dst_y in range(dst_height):
            for dst_x in range(dst_width):
                # 几何中心对称, 推导Z=0.5
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # # 向下取整, 作为左边相邻像素点坐标
                src_x1 = int(np.floor(src_x))
                # 右边相邻像素点坐标, 同时防止游标越界 (如果src_x1已经到了边界, 再+1就超出原图范围)
                src_x2 = min(src_x1 + 1, src_width - 1)
                src_y1 = int(np.floor(src_y))
                src_y2 = min(src_y1 + 1, src_height - 1)

                #         x2 - x              x - x1
                # y =   ------------ * y1 + ------------ * y2
                #         x2 - x1             x2 - x1
                # 由于是相邻像素点，分母化简取值为1
                # 且需要转换思维, 将y视为f(R1)求该点的【像素值】, 同理y1视为f(Q11)
                # 否则就是求src_y, 而src_y已经在上面通过几何中心对称求出, 显得没有意义

                # 综合公式 f()
                temp_a2 = (src_x2 - src_x) * img[src_y1, src_x1, i] + (src_x - src_x1) * img[src_y1, src_x2, i]
                temp_b2 = (src_x2 - src_x) * img[src_y2, src_x1, i] + (src_x - src_x1) * img[src_y2, src_x2, i]

                temp_a1 = src_y2 - src_y
                temp_b1 = src_y - src_y1

                dst_img[dst_y, dst_x, i] = temp_a1 * temp_a2 + temp_b1 * temp_b2

    return dst_img


if __name__ == '__main__':
    img = cv2.imread("./lenna.png")
    new_size = (768, 768)
    start = datetime.datetime.now()
    dst = bilinear(img, new_size)  # 耗时明显比最邻近插值要久
    end = datetime.datetime.now()
    print("func cost %s seconds" % (end - start).seconds)  # 15s 左右
    print(dst[20][16:20])
    cv2.imshow("bilinear interpolation", dst)

    # 直接调库CV2
    print("=" * 10)
    start = datetime.datetime.now()
    resized_image = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)  # resize默认使用双线性插值
    end = datetime.datetime.now()
    print("cv2 cost %s seconds" % (end - start).seconds)  # 0s
    print(resized_image[20][16:20])  # 边缘处像素差别较大, 中心处差别较小+-1 , 但是速度更快
    cv2.imshow("resized_image", resized_image)
    cv2.waitKey(0)
    print("=" * 10)

    # 直接调库PIL
    original_image = Image.open("./lenna.png")
    start = datetime.datetime.now()
    resized_image = original_image.resize(new_size, Image.Resampling.BILINEAR)
    end = datetime.datetime.now()
    print("PIL cost %s seconds" % (end - start).seconds)  # 0s
    print(resized_image)
    resized_image.show()

