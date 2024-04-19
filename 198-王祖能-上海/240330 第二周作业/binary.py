'''引入opencv,numpy,matplotlib等库'''
'''
@AUTHER ZENOWANG
彩色图像二值化
不同的读取方式cv2.imread或plt.imread结果也有不同；
转灰度方式cvtColor和rgb2gray也有不同，尽可能用同一个库；
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt  # 也可用import matplotlib.pyplot as plt
from skimage.color import rgb2gray

'''方法1：plt读取，rgb2gray转灰度化，plt绘图'''
figure = plt.imread('lenna.png')    # 通过cv2读取图片，是另外的结果，这里用plt读取
img_gray = rgb2gray(figure)     # 不同的库转灰度格式，结果也有区别
print(img_gray)
h, w = img_gray.shape
for i in range(h):
    for j in range(w):
        if img_gray[i, j] < 0.5:
            img_gray[i, j] = 0
        else:
           img_gray[i, j] = 1
# img_binary = np.where(img_gray < 0.5, 0, 1)       # 循环也可以用此式替代。
print(img_gray)
plt.imshow(img_gray, cmap='gray')
plt.show()
# cv2.imshow('demo', img_gray)      # 这里也可以采用cv2输出图像？？？图像结果相同
# cv2.waitKey(0)

'''方法2：plt读取，cv2.cvtColor转灰度化，plt绘图'''
figure = plt.imread('lenna.png')
img_gray = cv2.cvtColor(figure, cv2.COLOR_BGR2GRAY)
print(img_gray)
h, w = img_gray.shape
for i in range(h):
    for j in range(w):
        if img_gray[i, j] < 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
print(img_gray)
plt.imshow(img_gray, cmap='gray')
plt.show()

'''方法3：cv2读取，rgb2gray转灰度化，cv2绘图'''
figure = cv2.imread('lenna.png')
img_gray = rgb2gray(figure)
print(img_gray)
h, w = img_gray.shape
for i in range(h):
    for j in range(w):
        if img_gray[i, j] < 0.5:
            img_gray[i, j] = 0
        else:
           img_gray[i, j] = 1
print(img_gray)
cv2.imshow('demo3', img_gray)
cv2.waitKey(0)
# plt.imshow(img_gray, cmap='gray')     # 这里可以采用plt输出图像，图像结果相同
# plt.show()

'''方法4：cv2读取，cvtColor转灰度化，cv2绘图'''
figure = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(figure, cv2.COLOR_BGR2GRAY)    # cv2读取图片，并用cvtColor转灰度，结果是整数，要/255
img_gray = np.divide(img_gray, 255)
print(img_gray)
h, w = img_gray.shape
for i in range(h):
    for j in range(w):
        if img_gray[i, j] < 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
# img_binary = np.where(img_gray < 0.5, 0.0, 1.0) # 结果需要为0.0和1.0
print(img_gray)
cv2.imshow('demo4', img_gray)
cv2.waitKey(0)

