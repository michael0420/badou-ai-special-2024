import numpy as np
import cv2
import matplotlib.pyplot as plt

# 最近邻差值
def nearest_interp(img):
    height, width, channels = img.shape
    hn, wn = 700,700
    nearest_interp1 = np.zeros((hn, wn, channels), np.uint8)
    sh = hn / height
    sw = wn / width
    for i in range(hn):
        for j in range(hn):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            nearest_interp1[i, j] = img[x, y]
    return nearest_interp1


# 双线性差值
def bilinear_interp(img):
    src_h, src_w, channels = img.shape
    dst_h, dst_w = 700,700
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    bilinear_interp1 = np.zeros((dst_h, dst_w, channels), np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1 , src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                bilinear_interp1[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return bilinear_interp1


if __name__ == '__main__':
    img = cv2.imread('./lenna.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nearest_interp1 = nearest_interp(img)
    bilinear_interp1 = bilinear_interp(img)
    plt.subplot(221)
    plt.imshow(nearest_interp1)
    plt.subplot(222)
    plt.imshow(bilinear_interp1)
    plt.show()
    # cv2.imshow('nearest_interp1',nearest_interp1)
    # cv2.imshow('bilinear_interp1', bilinear_interp1)
    # cv2.waitKey(0)