# -*- coding: utf-8 -*-
'''
@ Author: Hayne
@ Time: 2024/04/04
@ Name: Bilinear Interpolation
'''

import numpy as np
import cv2


def bilinear_interpolation(img, outimg_shape):
    img_h, img_w, channel = img.shape
    outimg_h, outimg_w = outimg_shape[0], outimg_shape[1]
    if img_h == outimg_h and img_w == outimg_w:
        print("The shape of output image is same as the origin image!")
        return img.copy()
    out_img = np.zeros((outimg_h, outimg_w, channel), dtype=np.uint8)
    scale_x, scale_y = float(img_w) / outimg_w, float(img_h) / outimg_h

    for i in range(channel):
        for out_y in range(outimg_h):
            for out_x in range(outimg_w):
                src_x = (out_x + 0.5) * scale_x - 0.5
                src_y = (out_y + 0.5) * scale_y - 0.5

                src_x1 = int(np.floor(src_x))
                src_x2 = min(src_x1 + 1, img_w - 1)
                src_y1 = int(np.floor(src_y))
                src_y2 = min(src_y1 + 1, img_h - 1)

                R1 = (src_x2 - src_x) * img[src_y1, src_x1, i] + (src_x - src_x1) * img[src_y1, src_x2, i]
                R2 = (src_x2 - src_x) * img[src_y2, src_x1, i] + (src_x - src_x1) * img[src_y2, src_x2, i]
                out_img[out_y, out_x, i] = int((src_y2 - src_y) * R1 + (src_y - src_y1) * R2)
    return out_img


if __name__ == "__main__":
    img = cv2.imread("lenna.png")
    outimg_shape = [800, 800]
    out_img = bilinear_interpolation(img, outimg_shape)
    cv2.imshow('lenna', img)
    cv2.imshow('bilinear interp', out_img)
    cv2.waitKey()
