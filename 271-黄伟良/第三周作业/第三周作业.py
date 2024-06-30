import matplotlib.pyplot as plt
import numpy as np
import cv2

###双线性插值
def BilinearInterpolation(img, out_h, out_w):
    h, w, channel = img.shape
    out_img = np.zeros((out_h, out_w, channel), img.dtype)

    if out_w == w and out_h == h:
        return img.copy()

    scale_x = float(w) / out_w
    scale_y = float(h) / out_h

    for k in range(channel):
        for i in range(out_h):
            for j in range(out_w):
                x = (j + 0.5) * scale_x - 0.5
                y = (i + 0.5) * scale_y - 0.5

                x0 = int(np.floor(x))
                x1 = min(x0 + 1, w - 1)
                y0 = int(np.floor(y))
                y1 = min(y0 + 1, h - 1)

                temp1 = (x1 - x) * img[y0][x0][k] + (x - x0) * img[y1][x0][k]
                temp2 = (x1 - x) * img[y0][x1][k] + (x - x0) * img[y1][x1][k]

                out_img[i][j][k] = (y1 - y) * temp1 + (y - y0) * temp2
    return out_img


###直方图均衡化
def HistogramEqualization(img):

    h, w = img.shape[0:2]
    out_img = np.zeros([h, w], img.dtype)
    histInput = []

    for i in range(h):
        for j in range(w):
            flag = False
            for k in range(len(histInput)):
                if img[i][j] == histInput[k][0]:
                    histInput[k][1] = histInput[k][1] + 1
                    flag = True
            if flag == False:
                histInput.append([img[i][j], 1])

    histInput = sorted(histInput)

    for i in range(len(histInput)):

        Pi = histInput[i][1] / (h * w)

        if i == 0:
            sumPi = Pi
        else:
            sumPi = Pi + histInput[i-1][3]

        res = round(sumPi * 256 - 1)
        if res < 0 :
            res = 0

        histInput[i].append(Pi)
        histInput[i].append(sumPi)
        histInput[i].append(res)

    for i in range(h):
        for j in range(w):
            temp = img[i][j]

            for k in range(len(histInput)):
                if temp == histInput[k][0]:
                    out_img[i][j] = histInput[k][4]
    return out_img

###sobel边缘检测
def sobel(img, s):

    ###Kernel
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    h, w = img.shape[0:2]
    out_img = np.zeros([h, w], img.dtype)

    ###PADING
    p = int((3 - 1) / 2)
    padingImg = np.zeros([h + p * 2, w + p * 2], np.int16)


    for i in range(h):
        for j in range(w):
            padingImg[i + p][j + p] = img[i][j]

    ###卷积
    for i in range(h):
        for j in range(w):
            temp1 = padingImg[i:i+3, j:j+3]
            tempx = np.sum(temp1 * sobelX) * 0.5
            tempy = tempx + np.sum(temp1 * sobelY) * 0.5
            if abs(tempy) > 255 :
                out_img[i][j] = 255
            else:
                out_img[i][j] = abs(tempy)
    return out_img

gray_img = cv2.imread('lenna.png', 0)
RGB_img = cv2.imread('lenna.png',cv2.COLOR_BGR2RGB)
bil_img = BilinearInterpolation(RGB_img, 800, 800)
hist_img = HistogramEqualization(gray_img)
sobel_img = sobel(gray_img, 1)
cv2.imshow("bilImg", bil_img)
cv2.imshow("histImg", hist_img)
cv2.imshow("sobelImg", sobel_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
