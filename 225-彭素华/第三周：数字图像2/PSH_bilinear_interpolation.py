import numpy as np
import cv2


# 实现双线性插值

def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]

    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 中心对称
                src_x = (dst_x + 0.5) * scale_x - 0.5  # 公式
                src_y = (dst_y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))  # np.floor()返回不大于输入参数的最大证书。即：向下取整。int():实现将数据类型转化为整型。
                src_x1 = min(src_x0 + 1, src_w - 1)  # 这样是为了不超出图像大小的界限。
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 利用双线性插值的公式去计算双线性插值：
                tempo = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]  # i 表示的是通道信息
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int(src_y1 - src_y) * tempo + (src_y - src_y0) * temp1
    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png',1)
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()

