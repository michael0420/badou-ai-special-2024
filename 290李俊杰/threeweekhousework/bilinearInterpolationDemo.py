# 【第三周作业】
#
# 1. 实现双线性插值

import cv2
import numpy as np

# 1. 实现双线性插值
img = cv2.imread("lenna.png")
# 取出原图的512*512尺寸，外加通道数channel=3
srcH, srcW, c = img.shape
dstH = 900
dstW = 900
emptyimg = np.zeros((dstH, dstW, c), dtype=np.uint8)

# 遍历每一个通道进行插值
for i in range(3):
    for j in range(dstH):
        for s in range(dstW):
            # 先进行图像几何中心重合使坐标分配均匀
            srcX = (s + 0.5) * (srcW / dstW) - 0.5
            srcY = (j + 0.5) * (srcH / dstH) - 0.5
            # np.floor()返回不大于输入参数的最大整数。（向下取整）
            # 下面四行代码防止原图取的像素值坐标超出边界的防呆处理
            src_x0 = int(np.floor(srcX))  # np.floor()返回不大于输入参数的最大整数。（向下取整）
            src_x1 = min(src_x0 + 1, srcW - 1)
            src_y0 = int(np.floor(srcY))
            src_y1 = min(src_y0 + 1, srcH - 1)

            # 带入双线性插值公式求出输出图每个坐标上的像素值

            temp0 = (src_x1 - srcX) * img[src_y0, src_x0, i] + (srcX - src_x0) * img[src_y0, src_x1, i]
            temp1 = (src_x1 - srcX) * img[src_y1, src_x0, i] + (srcX - src_x0) * img[src_y1, src_x1, i]

            emptyimg[j, s, i] = int((src_y1 - srcY) * temp0 + (srcY - src_y0) * temp1)

# cv2.imshow("",img)
cv2.imshow("bilinear interp", emptyimg)
cv2.waitKey(0)
