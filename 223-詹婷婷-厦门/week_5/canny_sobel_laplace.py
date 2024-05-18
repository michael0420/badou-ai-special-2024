import cv2

from matplotlib import pyplot as plt


img = cv2.imread("lenna.png", 0)

img_sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) # 对x求导
img_sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) # 对y求导


img_laplace = cv2.Laplacian(img, cv2.CV_64F, ksize=3)

img_canny = cv2.Canny(img,100,150)
print(img_canny)

plt.subplot(231), plt.imshow(img, "gray"), plt.title("Original")
plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("Sobel_x")
plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("Sobel_y")
plt.subplot(234), plt.imshow(img_laplace,  "gray"), plt.title("Laplace")
plt.subplot(235), plt.imshow(img_canny, "gray"), plt.title("Canny")
plt.show()


