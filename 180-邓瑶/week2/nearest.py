# 最邻近插值法
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    cv2.imshow("one", img)

    height, width, channels = img.shape
    print(height, width, channels)

    img_dst = np.zeros((600, 600, channels), np.uint8)
    paramh = 600 / height
    paramw = 600 / width
    for i in range(600):
        for j in range(600):
            x = int(i / paramh + 0.5)
            y = int(j / paramw + 0.5)
            img_dst[i, j] = img[x, y]

    cv2.imshow("dest", img_dst)
    cv2.waitKey(0)
