import cv2
import matplotlib.pyplot as plot
import numpy as np


# 灰度直方图均衡化

if __name__ == '__main__':
    # 读入灰度图像
    gray_img = cv2.imread("image/test.jpg", cv2.IMREAD_GRAYSCALE)
    print(gray_img.shape)
    # 灰度直方图均衡化
    equalize_hist_img = cv2.equalizeHist(gray_img)
    # 画出灰度直方图，先原图后直方图
    cv2.imshow('Histogram Equalization', np.hstack([gray_img, equalize_hist_img]))
    plot.figure()
    plot.title('Grayscale Histogram')
    plot.xlabel('Bins')
    plot.ylabel('# of Pixels')
    plot.xlim([0, 256])
    plot.subplot(2, 1, 1)
    plot.hist(gray_img.ravel(), 256, [0, 255])
    plot.subplot(2, 1, 2)
    plot.hist(equalize_hist_img.ravel(), 256, [0, 255])
    plot.show()
    cv2.waitKey(0)
