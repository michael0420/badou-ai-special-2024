from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
cai_she_tu = cv2.imread("lenna.png")
cai_she_tu = cv2.cvtColor(cai_she_tu, cv2.COLOR_BGR2RGB)
cai_she_tu_to_gray = rgb2gray(cai_she_tu) #灰度化
print(cai_she_tu_to_gray)

r, c = cai_she_tu_to_gray.shape
for i in range(r):
    for j in range(c):
        if cai_she_tu_to_gray[i, j] <= 0.618:
            cai_she_tu_to_gray[i, j] = 0
        else:
            cai_she_tu_to_gray[i, j] = 1


plt.subplot(223)
plt.imshow(cai_she_tu_to_gray, cmap='gray')
plt.show()