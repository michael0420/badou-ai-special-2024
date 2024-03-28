import cv2
import numpy as np
import matplotlib.pyplot as plt

def nearest_interp(img):
    height, width, channel = img.shape
    new_image = np.zeros((800, 800, channel), dtype=img.dtype)
    sh = 800 / height  # 扩大或缩小的比例
    sw = 800 / width
    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)  ## int向下取整
            y = int(j / sw + 0.5)
            new_image[i, j] = img[x, y]

    return new_image


img = cv2.imread("lenna.png")
nearest_interp_result = nearest_interp(img)
cv2.imshow("nearest interp", nearest_interp_result)
cv2.imshow("image", img)
cv2.waitKey(0)


# img = plt.imread("lenna.png")
# nearest_interp_result = nearest_interp(img)
# plt.subplot(221)
# plt.imshow(nearest_interp_result)
# plt.subplot(222)
# plt.imshow(img)
# plt.show()  ## 竟然用Plt将原图和插值后的图放在一起看不出差别。
