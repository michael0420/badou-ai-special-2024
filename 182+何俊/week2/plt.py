#需要安装matplotlib包
#pip install matplotlib

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.color import rgb2gray

#用subpot输出原图
plt.subplot(221)
img = plt.imread(r"E:\AI\CV\second week\work\lenna.png")

plt.imshow(img)#是 Matplotlib 库中用于显示图像的函数，它接受多种参数以控制图像的显示方式


#用subpot输出灰度图
img = cv2.imread(r"E:\AI\CV\second week\work\lenna.png")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.subplot(222)
plt.imshow(img_gray,cmap = 'gray')
plt.show()
#plt.show() 是Matplotlib库中的一个函数，它的主要作用是将创建的图形显示在屏幕上，
#在非交互模式中如果在创建图形后没有调用 plt.show()，程序可能会在绘图命令之后立即退出，导致用户无法看到图形
