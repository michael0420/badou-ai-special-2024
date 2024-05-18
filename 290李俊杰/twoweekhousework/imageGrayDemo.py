# 【第二周作业】
#
# 1.实现灰度化和二值化  


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


# 实现灰度化
img=cv2.imread("lenna.png")
plt.subplot(221)
plt.imshow(img)
# 将img的像素值顺序改变成RGB
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
height,weight=img.shape[:2]
# 创建一个和原图一样大小的且像素值全为0 的新图片，img.dtype是像素值的数据类型
newimg=np.zeros([height,weight],img.dtype)
for i in range(height):
    for j in range(weight):
        # 获取原图矩阵第i行第j列的rgb像素值
        img_rgb=img[i,j]
        # 将像素值进行灰度化并赋值给新图片 gray=0.3*R+0.59*G+0.11*B
        # img_rgb保存的像素值是一个三维数组(R,G,B)
        newimg[i,j]=int(img_rgb[0]*0.3+0.59*img_rgb[1]+0.11*img_rgb[2])

plt.subplot(222)
plt.imshow(newimg,cmap="gray")
# print(newimg)
# cv2.imshow("image show gray",newimg)
# 原来，在运行cv2.imshow后，需要使用cv2.waitKey来保持窗口的显示
# cv2.waitKey(0)

# 实现二值化
img=cv2.imread("lenna.png")

h,w=img.shape[:2]
# 把图片转换成单通道的灰度图
img_binary=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("image show binarygray",img_binary)
# cv2.waitKey(0)
print(img_binary.shape)
for i in range(h):
    for j in range(w):
        # 对灰度图的每一个像素值除以255得到0-1的浮点值
        # 将小于0.5的像素值改成0，其余改成一得到二值图
        if img_binary[i,j]/255<=0.5:
            img_binary[i,j]=0
        else:
            img_binary[i, j] = 1
print(img_binary)
plt.subplot(223)
plt.imshow(img_binary, cmap='binary')
# cv2.imshow("image show binarygray",img_binary)
# cv2.waitKey(0)


# plt.subplot(221)
# img = plt.imread("lenna.png")
# # img = cv2.imread("lenna.png", False)
# plt.imshow(img)
#
#
# plt.subplot(222)
# gray_img=rgb2gray(img)
# plt.imshow(gray_img,cmap='gray')
#
#
#
# # 在网上查阅了一些资料发现，plt.imshow()这个函数
# # 的作用可以理解为：把一张图片做一些数据处理，但仅仅是做处理而已。
# #
# # 如果想要把图像显示出来，需要调用另外一个函数：plt.show()
plt.show()

