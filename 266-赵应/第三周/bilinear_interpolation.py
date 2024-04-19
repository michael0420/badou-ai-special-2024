import cv2
import numpy as np

# 双线性插值法


def nearest_interpolation(src_dim, out_dim):
    src_h, src_w, channel = src_dim.shape
    des_h, des_w = out_dim
    if src_h == des_h and src_w == des_w:
        return src_dim.copy()
    out_img = np.zeros((des_h, des_w, channel), np.uint8)
    rate_h = float(src_h) / des_h
    rate_w = float(src_w) / des_w

    for i in range(channel):
        for m in range(des_h):
            for n in range(des_w):
                src_x = (n + 0.5) * rate_w - 0.5
                src_y = (m + 0.5) * rate_h - 0.5

                src_x1 = int(np.floor(src_x))
                src_x2 = min(src_x1 + 1, src_w - 1)
                src_y1 = int(np.floor(src_y))
                src_y2 = min(src_y1 + 1, src_h - 1)
                # print(src_y1, src_y2, src_x1, src_x2)
                r1 = (src_x2 - src_x) * src_dim[src_y1, src_x1, i] + (src_x - src_x1) * src_dim[src_y1, src_x2, i]
                r2 = (src_x2 - src_x) * src_dim[src_y2, src_x1, i] + (src_x - src_x1) * src_dim[src_y2, src_x2, i]
                out_img[m, n, i] = int((src_y2 - src_y) * r1 + (src_y - src_y1) * r2)
    return out_img


if __name__ == '__main__':
    img = cv2.imread("image/lenna.png")
    target_img = nearest_interpolation(img, (750, 750))
    cv2.imshow('result', target_img)
    cv2.imshow('src', img)
    cv2.waitKey(0)
