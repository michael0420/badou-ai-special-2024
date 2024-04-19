
import cv2
import numpy as np


def bilinear_interpolation(img, dstWidth, dstHeight):

    srcW, srcH, channels = img.shape

    # 比例 后面目标图坐标映射至原图，就用目标坐标 * 比例，得到映射至原图的坐标
    w_scale = srcW / dstWidth
    h_scale = srcH / dstHeight

    new_img = np.zeros((dstWidth, dstHeight, channels), np.uint8)


    for dst_x in range(dstWidth):
        for dst_y in range(dstHeight):

            # 获取映射到原图的坐标，并且偏移了部分，偏移的目的是使2个图的几何中心相等
            src_x = (dst_x + 0.5) * w_scale - 0.5 # 0.5自己推导
            src_y = (dst_y + 0.5) * h_scale - 0.5

            # 找4个点的位置，找出后根据3次单线性插值求出目标【要注意边界条件】 (x0,y0) (x1,y0) (x0,y1) (x1,y1)
            src_x0 = int(np.floor(src_x))
            src_x1 = min(src_x0 + 1, srcW - 1) # 要减1，索引从0开始，不然到了下面的img[512]就会报错了
            src_y0 = int(np.floor(src_y))
            src_y1 = min(src_y0 + 1, srcH - 1)

            # 开始单线性插值
            temp0 = (src_x1 - src_x) * img[src_x0, src_y0] + (src_x - src_x0) * img[src_x1, src_y0]

            temp1 = (src_x1 - src_x) * img[src_x0, src_y1] + (src_x - src_x0) * img[src_x1, src_y1]

            new_img[dst_x, dst_y] = (src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1

    return new_img



img = cv2.imread('lenna.png')

dst = bilinear_interpolation(img, 700, 700)

cv2.imshow('bilinear interp',dst)
cv2.waitKey()