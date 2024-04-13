from skimage.color import rgb2gray
# 表示从 skimage.color 模块中导入名为 rgb2gray 的函数。rgb2gray 函数用于将 RGB 彩色图像转换为灰度图像，即将彩色图像的每个像素的红、绿、蓝三个通道的值进行加权平均，生成一个灰度值表示图像的亮度。
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


img=cv2.imread('lenna.png')
h,w=img.shape[:2]
img_gray=np.zeros([h,w],img.dtype)
for i in range (h):
    for j in range(w):
        m=img[i,j]
        img_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)

print(m)
#print(img_gray)
print(h,w)
#print("image show gray: %s"%img_gray)
cv2.imshow("image:show",img_gray)
cv2.waitKey()#用于防止闪退


plt.subplot(221) #设置子图位置2*2
img = plt.imread("lenna.png")
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
plt.show()  #不写 plt.show()不出图片
print("---image lenna----")
print(img)

# 灰度化
img_gray = rgb2gray(img)
cv2.imshow("dasdadsa",img_gray)
cv2.waitKey()
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)


#二值化

# rows,cols=img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if(img_gray[i,j]<=0.5):
#             img_gray[i,j]=0
#         else:
#             img_gray[i,j]=1

img_binary=np.where(img_gray<=0.5,0,1)
print("--------img_binary-------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()


