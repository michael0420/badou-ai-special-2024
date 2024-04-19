import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')
original_height,original_width,original_channels = img.shape
scale_height = int(input('请输入图片高度需要缩放的值：'))
scale_width = int(input('请输入图片宽度需要缩放的值：'))
scale_image = np.zeros([scale_height, scale_width, original_channels], img.dtype)
proportion_height = scale_height/original_height
proportion_width = scale_width/original_width
for i in range(scale_height):
    for j in range(scale_width):
        original_x = int(i/proportion_height)
        original_y = int(j/proportion_width)
        scale_image[i,j] = img[original_x,original_y]


print(scale_image)
# plt.subplot(221)
# plt.imshow(img,cmap="hot")
# plt.subplot(222)
# plt.imshow(change)
# plt.show()
cv2.imshow("scale_image",scale_image)
cv2.imshow("img",img)
cv2.waitKey(0)