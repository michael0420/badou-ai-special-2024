import cv2
import numpy as np
#导入图片
img = cv2.imread(r"E:\AI\CV\second week\work\lenna.png")
#以切片方式将图片宽高传给w和h
h,w = img.shape[:2]
#生成一个高宽为h,w，类型为和img同类型的数据，dtype用来获取数据类型
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
#imshow函数用来显示图像，在cv库中可以直接显示；在plt中需要用show来显示
cv2.imshow("gray",img_gray)
