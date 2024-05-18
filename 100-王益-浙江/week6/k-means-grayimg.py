
import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread('img/lenna.png', 0)
row, column = img.shape[:2]
data = img.reshape((row * column, 1))
data = np.float32(data)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
result = labels.reshape((row, column))
(plt.subplot(121),plt.imshow(img, cmap='gray'))
(plt.subplot(122),plt.imshow(result, cmap='gray'))
plt.show()
