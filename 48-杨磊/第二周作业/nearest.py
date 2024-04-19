import cv2
import numpy as np

# 最邻近插值
def nearestinterp(img):                       # 定义函数
    height,width,channels = img.shape         # 获取图片的长、宽、通道
    emptyImage = np.zeros((800,800,channels),np.uint8)   # 创建一张800*800和当前图像一样通道的图片
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh + 0.5)
            y=int(j/sw + 0.5)
            emptyImage[i,j]=img[x,y]          # 把计算的老图像像素给新图像赋值
    return emptyImage



img=cv2.imread("lenna.png")     # 读取图片
# zoom = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)   # 用函数放大图像
zoom=nearestinterp(img)         # 调用邻近插值函数处理
print(zoom)
print(zoom.shape)
cv2.imshow("image",img)         # 显示原图像
cv2.imshow("nearest",zoom)      # 显示放大后的图像
cv2.waitKey(0)
