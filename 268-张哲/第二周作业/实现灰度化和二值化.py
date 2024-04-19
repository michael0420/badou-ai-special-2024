import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./lenna.png')
image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print(img)
plt.subplot(221)
plt.imshow(image)
print("---image lenna----")
print(img)
def rgb_to_gray(img):
    h,w = img.shape[:2]
    im_gray = np.zeros([h,w],dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            m=img[i,j]
            im_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
    return im_gray

def gray_to_binary(img_gray):
    h,w = img_gray.shape[:2]
    im_binary = np.zeros([h,w],dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            if (img_gray[i,j]/255 <= 0.5):
                im_binary[i,j] = 0
            else:
                im_binary[i,j] = 1
    return im_binary

img_gray = rgb_to_gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)
img_binary = gray_to_binary(img_gray)
plt.subplot(223)
plt.imshow(img_gray, cmap='gray')
print("---image binary----")
print(img_binary)
plt.show()