#调用opencv数据库
import matplotlib.pyplot as plt
import numpy as np
import cv2

#灰度化
#读取图片
img = cv2.imread("C:/Users/86188/Pictures/lenna.png")
#获取图片的高度和宽度，切片
h,w = img.shape[:2]
#创建一张和当前图片大小一样的单通道图片
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        # 浮点整数算法,cv读入是BGR，没有进行转换直接是BGR
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
#img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)这串代码可以直接代替上面四行，直接将BGR三通道转化成了灰度图单通道

print(m)
#打印原始像素值
print(img_gray)
#打印灰度图
print("image show gray: %s"%img_gray)
#提示接下来将要输出的是灰度图像数据。
cv2.imshow("image show gray",img_gray)
#使用cv2.immshow函数显示灰度图


#二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
#np.where()函数接受三个参数，第一个参数是一个条件表达式，第二个参数是满足条件时的返回值，第三个参数是不满足条件时的返回值
print("-----imge_binary------")
#提示接下来将要输出的是二值化图像数据
print(img_binary)
print(img_binary.shape)


plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
#cmap参数表示颜色映射，'gray'表示使用灰度颜色映射。

plt.show()
#在调用imshow()函数显示图像后，需要调用show()函数才能真正显示图像







