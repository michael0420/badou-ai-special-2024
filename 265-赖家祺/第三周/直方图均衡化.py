import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./lenna.png")


def grayFunc():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 直方图均衡化
    dst = cv2.equalizeHist(gray)
    print(dst[:5])
    print("=" * 10)
    print(dst.ravel()[:5])

    # ravel()将多维数组转为一维数组
    plt.figure()
    plt.subplot(211)
    plt.hist(gray.ravel(), 256)  # 灰度图均衡化之前的直方图
    plt.subplot(212)
    plt.hist(dst.ravel(), 256)  # 灰度图均衡化之后的直方图
    plt.show()

    # 直方图，一维
    # hist = cv2.calcHist(dst, [0], None, [256], [0, 256])
    cv2.imshow("vertically compare", np.vstack([gray, dst]))  # 垂直堆叠

    cv2.waitKey(0)


def originFunc():
    (b, g, r) = cv2.split(img)
    blue_hist = cv2.equalizeHist(b)
    green_hist = cv2.equalizeHist(g)
    red_hist = cv2.equalizeHist(r)

    plt.figure()
    plt.subplot(321)
    plt.hist(blue_hist.ravel(), 256)

    plt.subplot(323)
    plt.hist(green_hist.ravel(), 256)

    plt.subplot(325)
    plt.hist(red_hist.ravel(), 256)

    # =========================
    plt.subplot(322)
    plt.hist(b.ravel(), 256)

    plt.subplot(324)
    plt.hist(g.ravel(), 256)

    plt.subplot(326)
    plt.hist(r.ravel(), 256)

    plt.show()

    result = cv2.merge((blue_hist, green_hist, red_hist))
    cv2.imshow("merged rbg", np.hstack([img, result]))

    cv2.waitKey(0)


if __name__ == '__main__':
    grayFunc()
    originFunc()
