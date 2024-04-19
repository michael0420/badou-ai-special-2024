import numpy as np
import cv2
import sys
import multiprocessing
import threading
flag_start = True

from matplotlib import pyplot as plt


def grayscale(img):#灰度化
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image


def binarize(img, threshold):#二值化
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图像
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def magnify(img,height,width):  #最邻近插值法放大

    h,w,c=img.shape

    empty_image = np.zeros((height,width,c),np.uint8)
    sh = height/h
    sw = width/w
    for i in range(height):
        for j in range(width):
            x = int(i/sh+0.5)
            y = int(j/sh+0.5)
            empty_image[i,j] = img[x,y]
    return empty_image

def bilinear_interpolation(img, new_height, new_width):
    height, width, channels = img.shape
    new_img = np.zeros((new_height, new_width, channels), np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            x = (i + 0.0) / new_height * height
            y = (j + 0.0) / new_width * width
            x1 = int(x)
            y1 = int(y)
            x2 = min(x1 + 1, height - 1)
            y2 = min(y1 + 1, width - 1)
            u = x - x1
            v = y - y1
            for k in range(channels):
                new_img[i, j, k] = (1 - u) * (1 - v) * img[x1, y1, k] + u * (1 - v) * img[x2, y1, k] + \
                                   (1 - u) * v * img[x1, y2, k] + u * v * img[x2, y2, k]
    return new_img


def histograme_qualization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
    plt.figure()
    plt.hist(dst.ravel(), 256)
    #plt.show()
    return gray,dst
def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv2.imshow("absX", absX)
    cv2.imshow("absY", absY)
    cv2.imshow("Result", dst)
    return absX,absY,dst

def start_thread():
        count = 0
        count += 1
        if count > 1 and cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 0:
            while True:
                user_input = input("Enter a value: ")
                if user_input.upper() == "STOP":
                    print('程序已终止！')
                    sys.exit()
                else:
                    thread = threading.Thread(target=stop, args=(user_input,))
                    thread.daemon = True  # 设置为守护线程，使得主线程结束时后台线程也会结束
                    thread.start()


while True:
    if flag_start:
        print('欢迎老师检查我的作业!')
    img = cv2.imread("lenna.png")
    print("0. EXIT")
    print("1. 灰度化插值法")
    print("2. 二值化插值法")
    print("3. 最邻近插值法放大法")
    print("4. 双线性插值放大法")
    print("5. 直方图均衡法")
    print("6. Sobel边缘检测法")
    try:
        method = int(input('请选择您的方法：'))
    except ValueError:
        print("输入错误，请输入一个整数作为方法选择")
        continue

    if   method == 0:
        flag_start = False
        break
    elif method == 1:
        flag_start = False
        print('您选择了灰度化插值法，请稍等！')
        gray_result = grayscale(img)
        cv2.imshow("Gray", gray_result)
        cv2.waitKey(0)
        continue
    elif method == 2:
        flag_start = False
        print('您选择了二值化插值法，请稍等！')
        threshold_value = 127  # 设置阈值
        binary_result = binarize(img, threshold_value)
        cv2.imshow("Binary", binary_result)
        cv2.waitKey(0)
        continue
    elif method == 3:
        flag_start = False
        print('您选择了最邻近插值法放大法，请稍等！')
        magnify_result = magnify(img, 800, 800)
        cv2.imshow("Magnify", magnify_result)
        cv2.waitKey(0)
    elif method == 4:
        flag_start = False
        print('您选择了双线性插值放大法，请稍等一会！')
        dst = bilinear_interpolation(img,700,700)
        cv2.imshow('Bilinear Interp', dst)
        cv2.waitKey(0)
        continue
    elif method == 5:
        flag_start = False
        print("您选择了直方图均衡法，请稍等一会！")
        gray,dst = histograme_qualization(img)
        cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
        plt.show()
        cv2.waitKey(0)
        continue
    elif method == 6:
        flag_start = False
        print("您选择了Sobel边缘检测法，请稍等一会！")
        sobel(img)
        cv2.waitKey(0)
    else:
        flag_start = False
        print("输入错误")
    cv2.waitKey(0)


cv2.waitKey(0)
cv2.destroyAllWindows()



