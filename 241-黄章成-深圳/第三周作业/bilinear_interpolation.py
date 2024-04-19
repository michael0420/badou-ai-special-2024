import cv2
import numpy


def bilinear_interpolation(image, outSize):
    src_h, src_w, c = image.shape
    dst_h, dst_w = outSize[1], outSize[0]

    if src_h == dst_h and src_w == dst_w:
        return image.copy()

    dst_image = numpy.zeros((dst_h, dst_w, c), dtype=numpy.uint8)
    sw, sh = float(src_w) / dst_w, float(src_h) / dst_h

    for i in range(c):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                # 找到对应原坐标点的位置
                # 为了使原坐标点系和目标点坐标系中心重合，需满足 src + 0.5 = (dst + 0.5) * s，s为放大/缩小比例
                src_x = (dst_x + 0.5) * sw - 0.5
                src_y = (dst_y + 0.5) * sh - 0.5

                # 找到(src_x, src_y)附近的两个点(src_x0, src_y0) 和 (src_x1, src_y1),同时注意边界问题
                src_x0 = int(numpy.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(numpy.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                p0 = (src_x1 - src_x) * image[src_y0, src_x0, i] + (src_x - src_x0) * image[src_y0, src_x1, i]
                p1 = (src_x1 - src_x) * image[src_y1, src_x0, i] + (src_x - src_x0) * image[src_y1, src_x1, i]
                dst_image[dst_y, dst_x, i] = int((src_y1 - src_y) * p0 + (src_y - src_y0) * p1)

    return dst_image


img = cv2.imread('lenna.png')
dst = bilinear_interpolation(img, (800, 800))
cv2.imshow('bilinear_interp', dst)
cv2.waitKey()