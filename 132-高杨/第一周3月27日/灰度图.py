import cv2
import matplotlib.pyplot as plt
import numpy as np



img = cv2.imread('lenna.png')

print(img.shape)

h,w=img.shape[:2]
imgGary=np.zeros([h,w],img.dtype)  # 创建单通道的图片



for i in range(h):
    for j in range(w):
        m=img[i,j]   # 原图是三通道，取得是 【216，165，234】
        imgGary[i,j]=int(m[0]*0.11 +m[1]*0.59 + m[2]*0.3)  #转化为单通道，每个像素点只有一个值

     #  m= twovalueimg[i,j]   #灰度图转化
      # print(m.shape)
       #m=int(0.11*m[0],0.59*m[1],0.3*m[2])
# img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 通过这个完成灰度化
print("image show gray:  %s" %imgGary)

cv2.imshow('twovalue',imgGary)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

plt.subplot(222)
plt.imshow(imgGary,cmap='gray')


plt.show()


