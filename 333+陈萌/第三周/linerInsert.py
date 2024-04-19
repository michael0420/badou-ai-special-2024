"""
双线性插值实现
"""
import numpy as np
import cv2


def linerInsert(img, out_dim):
    # 先获取原图像的h,w,c
    src_h, src_w, src_c = img.shape
    # 目标图像的h,w
    out_h, out_w = out_dim

    dst_img = np.zeros((out_h, out_w, src_c), dtype=img.dtype)

    # 计算缩放比例
    scale_h, scale_w = src_h / out_h, src_w / out_w
    for i in range(out_h):
        for j in range(out_w):
            # 获取目标图像的坐标
            # 保证原图像和目标图像的中心坐标相等 需要两边都加0.5
            x = (i + 0.5) * scale_h - 0.5
            y = (j + 0.5) * scale_w - 0.5
            # 负数的int是向上取整  所以要先 floor 向下取整 再进行整数转换
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = x0 + 1
            y1 = y0 + 1
            # 防止越界 x0 y0永远不会过界  因为到达最边界值时 -0.5会让该点在边界的左侧
            # print(x0)
            x0, x1 = x0, min(x1, src_h - 1)
            # x0, x1 = min(x0, src_h - 1), min(x1, src_h - 1)
            y0, y1 = y0, min(y1, src_w - 1)
            # 计算目标bgr值
            p0 = (x1 - x) * np.array(img[x0, y0]) + (x - x0) * np.array(img[x1, y0])
            p1 = (x1 - x) * np.array(img[x0, y1]) + (x - x0) * np.array(img[x1, y1])
            p = (y1 - y) * p0 + (y - y0) * p1
            dst_img[i, j] = list(p)

    return dst_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    out_dim = (700, 700)
    dst_img = linerInsert(img, out_dim)
    # 直接调用api进行双线性插值
    # dst_img = cv2.resize(img, out_dim, interpolation=cv2.INTER_LINEAR)
    print(dst_img)
    cv2.imshow('dst_img', dst_img)
    cv2.waitKey(0)