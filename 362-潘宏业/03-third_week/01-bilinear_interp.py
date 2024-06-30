import cv2
import numpy as np


def bilinear_interp(img, size):
    h, w, c = img.shape
    img_new = np.zeros((size[0], size[1], c), img.dtype)

    # 如果前后照片尺寸一样
    if h==size[0] and w==size[1]:
        return img

    # 缩放比例
    sh = float(h)/size[0]
    sw = float(w)/size[1]


    for i in range(img_new.shape[1]):
        for j in range(img_new.shape[0]):
            for k in range(img.shape[2]):
                # 中心对称
                src_x = (i + 0.5) * sw - 0.5
                src_y = (j + 0.5) * sh - 0.5

                # 目标点对应原图的四个位置
                x0 = int(src_x)
                x1 = min(x0+1, w-1)
                y0 = int(src_y)
                y1 = min(y0+1, h-1)

                # 代入公式
                temp0 = (y1 - src_y) * ((x1 - src_x) * img[y0, x0, k] + (src_x - x0) * img[y0, x1, k])
                temp1 = (src_y - y0) * ((x1 - src_x) * img[y1, x0, k] + (src_x - x0) * img[y1, x1, k])
                img_new[j, i, k] = int(temp0 + temp1)
    return img_new


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    img_new = bilinear_interp(img, [800, 800])
    cv2.imshow("initial", img)
    cv2.imshow("enlarge", img_new)
    cv2.waitKey()
