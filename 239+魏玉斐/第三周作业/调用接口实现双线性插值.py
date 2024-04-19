import cv2
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('lenna.png')
# 实现插值
# dsize是输出图像的大小，以元组形式表示，优先使用(宽度, 高度)的格式。
img_nearest = cv2.resize(img, (700, 700), interpolation=cv2.INTER_NEAREST)
img_resize = cv2.resize(img, (700, 700), interpolation=cv2.INTER_LINEAR)
img_cubic = cv2.resize(img, (700, 700), interpolation=cv2.INTER_CUBIC)
cv2.imshow('original', img)
cv2.imshow('inter_nearest', img_nearest)
cv2.imshow('inter_linear', img_resize)
cv2.imshow('inter_cubic', img_cubic)
cv2.waitKey(0)
