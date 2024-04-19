import cv2
import numpy as np


def bilinear(img, h, w):

    dst = np.zeros([h, w, 3], dtype=np.uint8)
    src_h, src_w, channel = img.shape

    if src_h == h and src_w == w:
        return img.copy()

    print(src_h)
    print(src_w)

    scale_x = float(src_w / w)
    scale_y = float(src_h / h)


    for c in range(channel):
        for y in range(h):
            for x in range(w):
                src_x = scale_x * (x + 0.5) - 0.5
                src_y = scale_y * (y + 0.5) - 0.5

                src_x1 = int(np.floor(src_x))
                src_x2 = min(src_x1 + 1, src_w - 1)
                src_y1 = int(np.floor(src_y))
                src_y2 = min(src_y1 + 1, src_h - 1)

                r1 = (src_x2 - src_x) * img[src_y1, src_x1, c] + (src_x - src_x1) * img[src_y2, src_x1, c]
                r2 = (src_x2 - src_x) * img[src_y1, src_x2, c] + (src_x - src_x1) * img[src_y2, src_x2, c]

                dst[y, x, c] = int((src_y2 - src_y) * r1 + (src_y - src_y1) * r2)
    return dst


if __name__ == '__main__':
    img = cv2.imread("1947.jpg")
    cv2.imshow("source", img)
    dst = bilinear(img, 1500, 2000)
    cv2.imshow("dst", dst)
    cv2.waitKey()


