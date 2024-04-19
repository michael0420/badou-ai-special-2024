import numpy as np
import cv2


def bilinear(img, out_dim):
    h, w, channel = img.shape
    h1, w1 = out_dim[1], out_dim[0]

    if h == h1 and w == w1:
        return img.copy()

    image_out = np.zeros((h1, w1, 3), dtype=np.uint8)

    # 图片比例
    scale_x, scale_y = float(w) / w1, float(h) / h1
    for i in range(channel):

        for dst_y in range(h1):
            for dst_x in range(w1):
                # 中心对称，目标图对应的原图坐标
                x = (dst_x + 0.5) * scale_x - 0.5
                y = (dst_y + 0.5) * scale_y - 0.5
                # 相邻四点坐标取整
                x0 = int(np.floor(x))
                y0 = int(np.floor(y))
                x1 = min(x0 + 1, w - 1)
                y1 = min(y0 + 1, h - 1)

                # 带入公式计算

                temp0 = (x1 - x) * img[y0, x0, i] + (x - x0) * img[y0, x1, i]
                temp1 = (x1 - x) * img[y1, x0, i] + (x - x0) * img[y1, x1, i]
                image_out[dst_y, dst_x, i] = int((y1 - y) * temp0 + (y - y0) * temp1)
    return image_out


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    image_out = bilinear(img, (700, 700))
    cv2.imshow('bilinear ', image_out)
    cv2.waitKey()
