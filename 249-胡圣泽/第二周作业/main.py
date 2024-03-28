import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image

#手动灰度化
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
#print(m)
#print(img_gray)
#print("image show gray :%s"%img_gray)
#cv2.imshow("image show gray",img_gray)
#cv2.waitKey(0)

img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#红色和蓝色的值互换
#cv2.imshow("image show RGB",img_RGB)
#cv2.waitKey(0)

img_gray2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("Gray2",img_gray2)
#cv2.waitKey(0)

#利用plt显示多张img，包括灰度化，二值化的图
#原图
plt.subplot(221)#221的含义是创建一张2*2大小的画布，当前img放在第一个位置
imag = plt.imread("lenna.png")
plt.imshow(imag)
print("----image lenna----")
print(imag)

#灰度化
imaggray = rgb2gray(imag)
plt.subplot(222)
plt.imshow(imaggray,cmap='gray')#必须加上后面cmap=gray，否则会用绿色来表示灰色
print("----imaggray----")
print(imaggray)

#二值化
#思路：灰度化的时候加入判断即可，太灰->黑色，靠近白->白色
height,width = imaggray.shape[:2]
imag_01 = np.zeros([height,width],imag.dtype)
for i in range(height):
    for j in range(width):
        #num = imaggray[i,j]/255，错误原因：观察灰度图的矩阵，灰度图已经是经过处理的0-1的浮点数了
        if imaggray[i,j] <= 0.5 :
            imag_01[i,j] = 0
        else:
            imag_01[i,j] = 1
plt.subplot(223)
plt.imshow(imag_01,cmap='gray')
print("----imag_01----")
print(imag_01)

plt.show()