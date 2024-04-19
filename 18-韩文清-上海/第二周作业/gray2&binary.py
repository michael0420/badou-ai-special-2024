from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#灰度化进阶版
img=cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #使用了OpenCV库（cv2模块）中的cvtColor函数，将图片从BGR颜色空间转换为灰度图
plt.subplot(211)
plt.imshow(img_gray, cmap='gray')
plt.title('image show gray')

#二值化
img_binary = np.where(img_gray >= 128, 255, 0)  #使用了NumPy库中的np.where()函数,利用条件表达式生成布尔数组
plt.subplot(212)
plt.imshow(img_binary, cmap='binary')
plt.title('image show binary')
plt.show()