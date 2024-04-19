import cv2
import numpy as np
import matplotlib.pyplot as plt
# 灰度化
img=cv2.imread('lenna.png')         #读图
h,w=img.shape[0:2]                  #获取图片的宽和高
img_gray=np.zeros([h,w],img.dtype)  #创建一张空白的新图片
for i in range(h):
    for j in range(w):
        z=img[i,j]
        img_gray[i,j]=int(z[0]*0.11+z[1]*0.59+z[2]*0.3)  # 灰度的计算
print(z)
print(img_gray)
print("image show gray:%s"%img_gray)
cv2.imshow("image show gray",img_gray)
cv2.imshow("image",img)
cv2.waitKey(0)

# 二值化
# r,c=img_gray.shape
# for i in range(r):
#     for i in range(c):
#         if (img_gray[i,j]/255 >=0.5):
#             img_gray[i,j]=0
#         else:
#             img_gray[i,j]=1
img_binary=np.where(img_gray/255 >=0.5,1,0)  #先归一化再比较
print(img_binary)
print(img_binary.shape)
plt.imshow(img_binary,cmap='gray')
plt.show()







