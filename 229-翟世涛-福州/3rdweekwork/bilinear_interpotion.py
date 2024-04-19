import cv2
import numpy as np


def bili_near_interpolation(img, num_h, num_w):
    h, w, c = img.shape
    target_img = np.zeros([num_h, num_w, c], np.uint8)

    for k in range(c):
        for j in range(num_h):
            for i in range(num_w):
                x = (i + 0.5) * w / num_w - 0.5
                y = (j + 0.5) * h / num_h - 0.5
                x0 = int(np.floor(x))
                x1 = min(x0 + 1, w - 1)
                y0 = int(np.floor(y))
                y1 = min(y0 + 1, w - 1)

                z0 = (x1 - x) * img[y0, x0, k] + (x - x0) * img[y0, x1, k]
                z1 = (x1 - x) * img[y1, x1, k] + (x - x0) * img[y1, x1, k]
                target_img[j, i, k] = int((y1 - y) * z0 + (y - y0) * z1)
    return target_img


if __name__ == '__main__':
    img = cv2.imread("../data/lenna.png")
    target_img = bili_near_interpolation(img, 1000, 1000)
    cv2.imshow("lenna", img)
    cv2.imshow("lenna_2", target_img)
    cv2.waitKey(0)
