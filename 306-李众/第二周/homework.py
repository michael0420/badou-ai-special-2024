import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

img = cv2.imread('/content/经济调控.png')
h,w,c = img.shape
# 灰度化
g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img, cmap=None)
plt.show()
plt.imshow(g_img, cmap='gray')
plt.show()
# 二值化
g_img = rgb2gray(img)
rows, cols = g_img.shape
for i in range(rows):
    for j in range(cols):
        if (g_img[i, j] <= 0.5):
            g_img[i, j] = 0
        else:
            g_img[i, j] = 1
g_img
plt.imshow(g_img, cmap='gray')
plt.show()
# 最临近插值
emptyImage=np.zeros((2800,5800,3),np.uint8)
hl = 2800/h
wl = 5800/w
for i in range(2800):
  for y in range(5800):
    x = int(i/hl+0.5)
    z = int(y/wl+0.5)
    if x>573:
      x=573
    if z>1199:
      z=1199
    emptyImage [i][y] = img[x][z]
plt.imshow(emptyImage, cmap='gray')
plt.show()
