import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray

# img = cv2.imread(r"C:\Users\lenovo\Desktop\lenna.png")
# h, w = img.shape[:2]
# img_gray = np.zeros((h, w), img.dtype)
# for i in range(h):
#     for j in range(w):
#         m = img[i, j]
#         img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

# print(img_gray)
# print("image show gray: %s"%img_gray)
# cv2.imshow('image show gray', img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


img_BGR = cv2.imread(r"C:\Users\lenovo\Desktop\lenna.png")
img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
plt.figure()
plt.subplot(2,2,1)
plt.imshow(img)


img_gray = rgb2gray(img)
plt.subplot(2,2,2)
plt.imshow(img_gray, cmap = 'gray')

img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(2,2,3)
plt.imshow(img_binary, cmap = 'gray')
plt.show()
