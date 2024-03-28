import cv2
import numpy as np

#读取原图
org_img  = cv2.imread("../../lenna.png")

#创建900*900的上采样图片
h , w, c = org_img.shape
new_img = np.zeros((900,900,c),dtype=np.uint8)

#计算缩放比例
sh = 900 / h
sw = 900 / w

#遍历新图的每一个像素点坐标
for i in range(900):
    for j in range(900):
        x = int(i / sh +0.5)   #计算目前的i值和原图的哪个x点最近，获取到该x的坐标
        y = int(j / sw +0.5)
        new_img[i,j] = org_img[x,y]   #获取原图的像素点值，赋值给新图的坐标

cv2.imshow("org_img",org_img)
cv2.imshow("new_img",new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()











