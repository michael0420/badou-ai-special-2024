"""
实现双线性插值
"""

import cv2
import numpy as np

def bilinear_interpolation(img,out_dim):
    src_h,src_w,channel = img.shape
    dst_h,dst_w = out_dim[1],out_dim[0]
    # 若尺寸一样则返回原图
    if src_h == dst_h and src_w == dst_h:
        return img.copy()
    dst_img = np.zeros([dst_h,dst_w,channel],dtype=np.uint8)
    scale_x,scale_y = float(src_w)/float(dst_w),float(src_h)/float(dst_h)
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                #对齐中心
                src_x = (dst_x + 0.5)*(scale_y) - 0.5
                src_y = (dst_y + 0.5)*(scale_x) - 0.5

                #防止越界
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                #计算双线性插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


img = cv2.imread("lenna.png")
dst = bilinear_interpolation(img,(700,700))
cv2.imshow('bilinear interp',dst)
cv2.waitKey()