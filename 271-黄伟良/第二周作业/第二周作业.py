import cv2
import numpy as np
import matplotlib.pyplot as plt

###灰度化###
def rgb_to_gray(weight,height,img):

    grayImg1 = np.zeros([height,weight],img.dtype)
    grayImg2 = np.zeros([height,weight],img.dtype)
    grayImg3 = np.zeros([height,weight],img.dtype)

    #1、使用浮点算法：Gray = 0.3*R + 0.59*G + 0.11*B
    for i in range(h):
        for j in range(w):
            R = img[i][j][0]
            G = img[i][j][1]
            B = img[i][j][2]

            Gray_1 = 0.3*R + 0.59*G + 0.11*B
            grayImg1[i][j] = Gray_1

    #2、使用平均值法
    for i in range(h):
        for j in range(w):
            R = img[i][j][0]
            G = img[i][j][1]
            B = img[i][j][2]

            Gray_2 = (R + G + B) / 3
            grayImg2[i][j] = Gray_2

    #3、使用G值法
    for i in range(h):
        for j in range(w):
            G = img[i][j][1]

            Gray_2 = G
            grayImg3[i][j] = Gray_2
    cv2.imshow("Nearest", grayImg1)
    cv2.waitKey(0)

    return grayImg1

###二值化###
def RGB_to_Binary(h,w,grayImg):
    binaryImg = np.zeros([h,w],img.dtype)
    grayImg = grayImg / 255
    print(grayImg)
    for i in range(h):
        for j in range(w):
            if grayImg[i][j] <= 0.5:
                binaryImg[i][j] = 0
            else:
                binaryImg[i][j] = 1

    plt.imshow(binaryImg,cmap='gray')
    plt.show()

###最邻近插值法###
def Nearest_Interp(h,w,img,h1=1026,w1=1280):
    emptyImg = np.zeros([h1,w1,3],img.dtype)

    eachRows = h1 / h
    eachCols = w1 / w

    for i in range(h1):
        for j in range(w1):
            newRowIndex = int(i / eachRows + 0.5)
            newColIndex = int(j / eachCols + 0.5)
            if newRowIndex >= h:
                newRowIndex = h - 1
            if newColIndex >= w:
                newColIndex = w - 1
            emptyImg[i][j] = img[newRowIndex][newColIndex]

    cv2.imshow("Nearest",emptyImg)
    # cv2.waitKey(0)

img = cv2.imread('lenna.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
h,w = img.shape[:2]
grayImg = rgb_to_gray(w,h,img)

RGB_to_Binary(h,w,grayImg)

Nearest_Interp(h,w,img)