import cv2
import numpy as np
import matplotlib.pyplot as plt

def gray_equalhist(file):
    img = cv2.imread(file, 0)
    # 灰度图像的直方图
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0,256])
    plt.show()
    # 灰度图像直方图均衡化
    dst = cv2.equalizeHist(img)
    plt.figure()
    plt.hist(dst.ravel(), 256)
    plt.show()

    cv2.imshow("Histogram Equalization", np.hstack([img, dst]))
    cv2.waitKey(0)

def color_equalhist(file):
    img = cv2.imread(file, 1)
    cv2.imshow("src", img)
    b, g, r = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    
    result = cv2.merge((bH, gH, rH))
    cv2.imshow("dst", result)
    cv2.waitKey(0)

if __name__ == "__main__":
    file = "lenna.png"
    gray_equalhist(file)
    color_equalhist(file)