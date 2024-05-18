
import cv2
import numpy as np

# 读取图像
img = cv2.imread('lenna.png')

# 定义源图像和目标图像的尺寸
h, w = img.shape[:2]
new_w, new_h = 700, 700

# 创建x和y方向上的映射矩阵
# 这里的映射矩阵是根据新的尺寸来创建的
map_x = np.zeros((new_h, new_w, 1), dtype=np.float32)
map_y = np.zeros((new_h, new_w, 1), dtype=np.float32)

for y in range(new_h):
    for x in range(new_w):
        map_x[y, x] = (x + 0.5) * (w / new_w)
        map_y[y, x] = (y + 0.5) * (h / new_h)

# 使用双线性插值方法进行重采样
resized_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

# 保存结果
# cv2.imwrite('output.jpg', resized_img)
cv2.imshow('bilinear interp',resized_img)
cv2.waitKey()
