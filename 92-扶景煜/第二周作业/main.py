import numpy
import cv2

def grayscale(img):#灰度化
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image

def binarize(img, threshold):#二值化
    _, binary_image = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def magnify(img,height,width):#最邻近插值法放大
    h,w,c=img.shape
    emptyImage=numpy.zeros((height,width,c),numpy.uint8)
    sh=height/h
    sw=width/w
    for i in range(height):
        for j in range(width):
            x = int(i/sh+0.5)
            y = int(j/sh+0.5)
            emptyImage[i,j] = img[x,y]
    return emptyImage

img = cv2.imread("lenna.png")

gray_result = grayscale(img)
cv2.imshow("Gray Image", gray_result)

threshold_value = 127  # 设置阈值
binary_result = binarize(gray_result, threshold_value)
cv2.imshow("Binary Image", binary_result)
#如果用初始图片进行二值化会得到彩色的二值化图片，即对三个通道分别进行二值化，否则得到灰白色的，即单通道
magnify_result = magnify(img,800,800)
cv2.imshow("new Image",magnify_result)

cv2.waitKey(0)
