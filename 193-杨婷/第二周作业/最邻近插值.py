"""
@author 193-杨婷
最邻近插值
"""
import cv2
import numpy as np


# 手撸
def function(img):
    height, width, channels = img.shape  # (512,512,3)
    emptyImage = np.zeros((800, 800, channels), np.uint8)  # 放大图像
    sh = 800/height  # 缩放比例
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img = cv2.imread("lenna.png")
zoom = function(img)
print(zoom)
print("--------------------------------------")
print(zoom.shape)

cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)

# 调用方法
"""
interpolation 是插值方法，用于在调整大小时处理图像像素值。
常用的插值方法包括：cv2.INTER_LINEAR（线性插值，默认值）
cv2.INTER_NEAREST（最近邻插值）
cv2.INTER_AREA（区域插值，适用于缩小图像）
cv2.INTER_CUBIC（双三次插值，高质量放大）
"""
zoom1 = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)
print(zoom1)
print("--------------------------------------")
print(zoom1.shape)
cv2.imshow("nearest interp", zoom1)
cv2.imshow("image", img)
cv2.waitKey(0)

