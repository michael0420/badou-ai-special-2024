# 实现二值化

# 1- import packages
import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage.color import rgb2gray

# 2- import images and convert to grayscale
img = cv2.imread("../images/lenna.png")
h,w = img.shape[:2]
# img_gray = rgb2gray(img) --> method One, better than method two.
# method two - when using cv2.cvtColor, need to normalize it by /255.0 to convert values into range btw 0-1
img_gray = ( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ) / 255.0


plt.subplot(221)
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale image')

# 3.1 binary - manually
for i in range(h):
    for j in range(w):
        if img_gray[i,j] <= 0.5:
            img_gray[i,j] = 0
        else:
            img_gray[i,j] = 1

img_binary_one = img_gray

plt.subplot(222)
plt.imshow(img_binary_one, cmap='gray')
plt.title('Binary - manually')

# 3.2 - binary via python function np.where
img_binary_two = np.where(img_gray >= 0.5, 1, 0)

plt.subplot(223)
plt.imshow(img_binary_two, cmap='gray')
plt.title('Binary by function')
plt.tight_layout()
plt.savefig('Binary images.png')
plt.show()






