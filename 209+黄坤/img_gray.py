from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread("img.png")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1] * 0.59 + m[2] * 0.3)
print(img_gray)
print("image show gray is : %s"%img_gray)
cv2.imshow("image show gray",img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
#imshow显示图片需要调用waitKey来保持窗口显示，或者可以cv2.imwrite('src.jpg', src_im)将图片输出

plt.subplot(221)
img = plt.imread("img.png")
plt.imshow(img)
print("---image img ---")
print(img)