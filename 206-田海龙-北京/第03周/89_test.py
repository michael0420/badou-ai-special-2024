
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
 
img = cv.imread(r"D:\Temp\Test\lenna.png", cv.IMREAD_COLOR)
print(img.shape)

hist_b = cv.calcHist([img], [0], None, [256], [0,256])
hist_g = cv.calcHist([img], [1], None, [256], [0,256])
hist_r = cv.calcHist([img], [2], None, [256], [0,256])

#显示图像
fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10), dpi=100)
# 第三个维度调换顺序
axes[0][0].imshow(img[:,:,::-1])
axes[0][0].set_title("Original")
axes[0][1].set_title("Original Histogram")
axes[0][1].plot(hist_b, color='b')
axes[0][1].plot(hist_g, color='g')
axes[0][1].plot(hist_r, color='r')


def RGBImageHistEqualize(src):
    src=src[:,:,::-1]
    # plt.figure()
    # plt.imshow(src)

    hsv = cv.cvtColor(src, cv.COLOR_RGB2HSV)
    

    # plt.figure()
    # plt.imshow(hsv)
    
    channels = cv.split(hsv)
    #对亮度通道进行直方图均衡化
    cv.equalizeHist(channels[2], channels[2])
    hsv = cv.merge(channels)
    print(hsv.shape)
   
    src = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

    # plt.figure()
    # plt.imshow(src)

    return src

img_hist_equalized = RGBImageHistEqualize(img)
hist_equalized_r = cv.calcHist([img_hist_equalized], [0], None, [256], [0,256])
hist_equalized_g = cv.calcHist([img_hist_equalized], [1], None, [256], [0,256])
hist_equalized_b = cv.calcHist([img_hist_equalized], [2], None, [256], [0,256])


#显示直方图均衡化后的结果
axes[1][0].imshow(img_hist_equalized)
axes[1][0].set_title("Equalized")
axes[1][1].set_title("Equalized Histogram")
axes[1][1].plot(hist_equalized_b, color='b')
axes[1][1].plot(hist_equalized_g, color='g')
axes[1][1].plot(hist_equalized_r, color='r')

plt.show()





