# 灰度化
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(221)
    plt.imshow(img, cmap="gray")

    # 灰度化
    h, w, c = img.shape
    img_gray = np.zeros([h, w], img.dtype)
    img_gray2 = np.zeros([h, w], img.dtype)
    print(h, w, c)
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            img_gray[i, j] = int(0.3 * m[0] + 0.59 * m[1] + 0.11 * m[2])

    plt.subplot(222)
    plt.imshow(img_gray, "gray")

    # 二值化
    img_2 = np.where(img_gray / 255 > 0.5, 1, 0)
    plt.subplot(224)
    plt.imshow(img_2, "gray")
    plt.show()
