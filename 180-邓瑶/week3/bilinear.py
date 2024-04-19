# 双线性插值法
import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("../lenna.png")
    # cv2.imshow("ori", img)

    height, width, channels = img.shape

    img_dst = np.zeros((600, 600, channels), np.uint8)

    for c in range(channels):
        for h in range(600):
            for w in range(600):
                # 取点
                x = max((w + 0.5) * float(width) / 600 - 0.5, 0)
                y = max((h + 0.5) * float(height) / 600 - 0.5, 0)

                # 取邻近四点
                x0 = int(np.floor(x))
                x1 = min(x0 + 1, width -1)
                y0 = int(np.floor(y))
                y1 = min(y0 + 1, height -1)

                # 取邻近四点的值
                f1 = img[y0, x0, c]
                f2 = img[y0, x1, c]
                f3 = img[y1, x0, c]
                f4 = img[y1, x1, c]

                img_dst[h, w, c] = (y1 - y) * ((x1 - x) * f1 + (x - x0) * f2) + (y - y0) * ((x1 - x) * f3 + (x - x0) * f4)

    cv2.imshow("dest", img_dst)
    cv2.waitKey(0)