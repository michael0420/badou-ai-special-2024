import cv2 as cv
import numpy as np


def bilinear_interpolation(img, out_dim):
    src_h, src_w, c = img.shape
    dst_h, dst_w = out_dim[0], out_dim[1]
    if dst_h == src_h and dst_w == src_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, c), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(c):
        for y in range(dst_h):
            for x in range(dst_w):
                # 查找对原图像的坐标点
                # 几何中心对称
                src_x = (x + 0.5) * scale_x - 0.5
                src_y = (y + 0.5) * scale_y - 0.5

                x1 = int(src_x)
                y1 = int(src_y)
                x2 = min(x1 + 1, src_w - 1)
                y2 = min(y1 + 1, src_h - 1)

                temp1 = (x2 - src_x) * img[y1, x1, i] + (src_x - x1) * img[y1, x2, i]
                temp2 = (x2 - src_x) * img[y2, x1, i] + (src_x - x1) * img[y2, x2, i]
                dst_img[y, x, i] = int((y2 - src_y) * temp1 + (src_y - y1) * temp2)
    return dst_img


if __name__ == '__main__':
    img = cv.imread('../lenna.png')
    out_dim = (700, 700)
    dst = bilinear_interpolation(img, out_dim)
    cv.imshow("dst", dst)
    cv.waitKey(0)
