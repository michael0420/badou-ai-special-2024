# 练习图片最临近插值
import cv2
import numpy as np

def func(img):
    height, width, channel = img.shape
    emptyImage = np.zeros((800, 800, channel), np.uint8)  # 创建一张尺寸为800 * 800的空白图片
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh + 0.5) # int()，转为整数，向下取整
            y = int(j/sw + 0.5)
            emptyImage[i, j] = img[x,y] # 遍历空白图片的每个坐标，并将img的x，y坐标处的像素值赋值给空白图片
    return emptyImage # 函数返回图片文件

img = cv2.imread("lenna.png") # 获取图片对象
emptyImage = func(img)
print(emptyImage)
print(emptyImage.shape)
cv2.imshow("img", img)
cv2.imshow("emptyImage", emptyImage)
cv2.waitKey(0)