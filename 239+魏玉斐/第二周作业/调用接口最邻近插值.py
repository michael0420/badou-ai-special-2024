import cv2

img = cv2.imread('lenna.png')
# 实现最邻近插值
# dsize,这是输出图像的大小，以元组形式表示，优先使用(宽度, 高度)的格式。
resized = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)
cv2.imshow('Original', img)
cv2.imshow('Nearest_interp', resized)
cv2.waitKey(0)
