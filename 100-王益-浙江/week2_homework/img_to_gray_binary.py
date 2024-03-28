import cv2
import numpy as np
import matplotlib.pyplot as plt
def img_to_gray(img):
    height, width = img.shape[:2]
    img_gray = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            img_gray[i, j] = img[i, j, 0] * 0.11 + img[i, j, 1] * 0.59 + img[i, j, 2] * 0.3
    return img_gray

def img_to_binary(img_gray):
    img_binary = np.where(img_gray >= 128, 255, 0).astype(np.uint8)
    return img_binary

img = cv2.imread("img/fish.jpg")
img_gray = img_to_gray(img)
img_binary = img_to_binary(img_gray)
plt.subplot(221)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()

img= cv2.imread("img/lenna.png")
img_gray = img_to_gray(img)
img_binary = img_to_binary(img_gray)
cv2.imshow("Original Image", img)
cv2.imshow("Gray Image", img_gray)
cv2.imshow("Binary Image", img_binary)
cv2.waitKey(0)