# 双线性插值
import cv2
import numpy as np
img = cv2.imread("lenna.png")
src_h , src_w , src_channel = img.shape
dst_img = np.zeros((800,800,3), img.dtype)
dst_h , dst_w , dst_channel = dst_img.shape
nh, nw= dst_h/src_h , dst_w/src_w
for i in range(dst_channel):
    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            # 中心对齐
            src_x = (dst_x + 0.5)/ nw-0.5
            src_y = (dst_y + 0.5)/ nh-0.5
            # 找插入点周围最相近的四个点
            src_x1 = int(np.floor(src_x))
            src_x2 = min(src_x1 + 1, src_w - 1)
            src_y1 = int(np.floor(src_y))
            src_y2 = min(src_y1 + 1, src_h - 1)
            # 代入公式
            dst_img[dst_x,dst_y,i] = int((src_y2-src_y)*((src_x2-src_x)*img[src_x1,src_y1,i]+(src_x-src_x1)*img[src_x2,src_y1,i]) + (src_y-src_y1)*(((src_x2-src_x)*img[src_x1,src_y2,i]+(src_x-src_x1)*img[src_x2,src_y2,i])))

cv2.imshow("bilinear interpolation", dst_img)
cv2.waitKey()
