import cv2
import numpy as np

# 读取图片
img = cv2.imread('lenna.png')
# 获取图片高、宽
h, w, c = img.shape
# 创建一张新的放大图片
img_bigger = np.zeros([800, 800, c], np.uint8)
# 获取图片缩放比例
y_ratio = h / 800
x_ratio = w / 800
for i in range(c):
    for y in range(800):
        for x in range(800):
            # 计算放大图片像素点在原图的位置(浮点数）
            src_x = (x + 0.5) * x_ratio - 0.5
            src_y = (y + 0.5) * y_ratio - 0.5
            # 获得原图该节点附近的四个点的坐标
            src_x0 = int(src_x)
            src_x1 = min(src_x0 + 1, w - 1)
            src_y0 = int(src_y)
            src_y1 = min(src_y0 + 1, h - 1)

            temp_x0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
            temp_x1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
            img_bigger[y, x, i] = int((src_y1 - src_y) * temp_x0 + (src_y - src_y0) * temp_x1)

print(img_bigger)
cv2.imshow("bigger", img_bigger)
cv2.waitKey(0)

