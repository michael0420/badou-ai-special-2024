import cv2
import numpy as np

def bilinear_interpolation(img):
    dst = np.zeros((1000, 1000, 3), dtype=np.uint8)
    height, width, channel = img.shape
    factor_h = float(width-1)/(1000-1)
    factor_w = float(width-1)/(1000-1)
    for i in range(1000):
        for j in range(1000):
            dst_h = i * factor_h
            dst_w = j * factor_w
            w0, h0 = int(dst_w), int(dst_h)
            w1, h1 = w0+1, h0+1
            if w0 == width-1:
                w0 = width-2
                w1 = width-1
            if h0 == height-1:
                h0 = height-2
                h1 = height-1
            p0 = img[w0, h0] * (w1-dst_w) + img[w1, h0] * (dst_w - w0)
            p1 = img[w0, h1] * (w1-dst_w) + img[w1, h0] * (dst_w - w0)
            dst[j, i] = p0 * (h1-dst_h) + p1 * (dst_h-h0)

    return dst


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
