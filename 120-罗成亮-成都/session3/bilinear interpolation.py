import cv2
import numpy as np


def quadratic_interpolation(img, proportion):
    if proportion == 1:
        return img.copy()
    height, width, channel = img.shape
    dst_height = proportion * height
    dst_width = proportion * width

    dst_img = np.zeros((dst_height, dst_width, channel), dtype=np.uint8)

    for c in range(channel):
        for dst_y in range(dst_height):
            for dst_x in range(dst_width):
                src_x = (dst_x + 0.5) / proportion - 0.5
                src_y = (dst_y + 0.5) / proportion - 0.5

                src_x0 = int(src_x)
                src_x1 = min(src_x0 + 1, width - 1)
                src_y0 = int(src_y)
                src_y1 = min(src_y0 + 1, height - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, c] + (src_x - src_x0) * img[src_y0, src_x1, c]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, c] + (src_x - src_x0) * img[src_y1, src_x1, c]

                dst_img[dst_y, dst_x, c] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img


if __name__ == '__main__':
    img = cv2.imread("../lenna.png")
    dst_img = quadratic_interpolation(img, 2)
    cv2.imshow('quadratic interpolation', dst_img)
    cv2.waitKey(0)
