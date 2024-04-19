'''
作业1：
实现双线性插值
'''
import cv2
import numpy as np


# 定义一个方法实现双线性插值
def bilinear_interp(img, out_dim):

    h, w, c = img.shape
    out_h, out_w = out_dim[0], out_dim[1]

    if h == out_h and w == out_w:
        return img.copy()

    out_img = np.zeros((out_h, out_w, c), dtype=np.uint8)       # 创建放大后图像的矩阵
    scale_x, scale_y = float(w)/out_w, float(h)/out_h           # 求图像各边放大的比率

    for i in range(c):
        for out_x in range(out_h):
            for out_y in range(out_w):                          # 遍历各通道下的每一个像素值

                src_x = (out_x + 0.5) * scale_x - 0.5           # 求出输出矩阵于输入矩阵的对应关系
                src_y = (out_y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))                   # 求出虚拟像素点最近的四个像素点的位置，
                src_x1 = min(src_x0 + 1, w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, h - 1)
                                                                # 通过双线插值，求出虚拟像素点的值
                temp0 = (src_x - src_x0)*img[src_x1, src_y0, i] + (src_x1 - src_x)*img[src_x0, src_y0, i]
                temp1 = (src_x - src_x0)*img[src_x1, src_y1, i] + (src_x1 - src_x)*img[src_x0, src_y1, i]
                out_img[out_x, out_y, i] = (src_y - src_y0) * temp1 + (src_y1 - src_y) * temp0

    return out_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img_interp = bilinear_interp(img, (1024, 1024))
    cv2.imshow('img', img)
    cv2.imshow('bilinear_interpolation_img', img_interp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
