import numpy as np
import cv2

img = cv2.imread('lenna.png')
h, w, c = img.shape
dst_h, dst_w = 1000, 1000
img_dst = np.zeros((dst_h, dst_w, c), dtype=np.uint8)
#计算缩放比例
ratio_y = float(h/dst_h)
ratio_x = float(w/dst_w)
if (dst_h,dst_w) == (h, w):
    img_dst = img.copy
else:
    for dst_x in range(dst_w):
        for dst_y in range(dst_h):
            # 计算中心对齐的原坐标  都移动0.5使中心对齐
            src_x = (dst_x+0.5)*ratio_x-0.5
            src_y = (dst_y+0.5)*ratio_y-0.5
            # 计算相邻的坐标点
            src_x1 = int(np.floor(src_x))
            src_x2 = min(src_x1+1, w-1)
            src_y1 = int(np.floor(src_y))
            src_y2 = min(src_y1+1, h-1)
            # X(w)方向做插值
            # f(R1) = f(x,y1) = (x2-x)f(Q11) + (x - x1)f(Q21)
            # f(R2) = f(x,y2) = (x2-x)f(Q12) + (x - x1)f(Q22)
            dst1 = (src_x2-src_x)*img[src_y1,src_x1]+(src_x-src_x1)*img[src_y1,src_x2]
            dst2 = (src_x2-src_x)*img[src_y2,src_x1]+(src_x-src_x1)*img[src_y2,src_x2]
            # Y(h)方向做插值 f(x,y) = (y2-y)f(R1)+(y1-y)f(R2)
            img_dst[dst_y,dst_x] = (src_y2-src_y)*dst1 + (src_y-src_y1)*dst2
cv2.imshow('lenna',img)
cv2.imshow('bilinear_interp',img_dst)
cv2.waitKey(0)
