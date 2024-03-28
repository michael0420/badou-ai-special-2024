import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读图与创建空图
img = cv2.imread('lenna.png')
h, w, channels = img.shape
h_wanted = 800
w_wanted = 800
sh = h_wanted / h
sw = w_wanted / w
img_result = np.zeros((h_wanted, w_wanted, channels), dtype=np.uint8)
# cv2.imshow('image_empty', img_result)
# cv2.waitKey(0)

# 临近插值法
for i in range(h_wanted):
    for j in range(w_wanted):
        x = int(i / sh + 0.5)
        y = int(j / sw + 0.5)
        img_result[i, j] = img[x, y]
cv2.imshow('origin', img)
cv2.imshow('result', img_result)
cv2.waitKey(0)

# plt显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(img)
img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
plt.subplot(122)
plt.imshow(img_result)
plt.show()
