import cv2
import numpy as np


def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    # 不转换回uint8就只是显示灰色窗口
    absx = cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)

    # cv2.imshow("x sobel", absx)
    # cv2.imshow("y sobel", absy)

    dst = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

    # cv2.imshow("dst", dst)
    cv2.imshow("dst", np.hstack([absx, absy, dst]))

    cv2.waitKey(0)


if __name__ == '__main__':
    img = cv2.imread("./lenna.png", cv2.IMREAD_COLOR)
    sobel(img)
    img = cv2.imread("./lenna.png", cv2.IMREAD_GRAYSCALE)
    sobel(img)
