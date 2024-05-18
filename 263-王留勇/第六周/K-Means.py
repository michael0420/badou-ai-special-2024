import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

rows, cols = gray.shape
data = gray.reshape((rows * cols, 1))
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
dst = labels.reshape(gray.shape)

titles = ['src img', 'kmeans img']
images = [gray, dst]

for i in range(len(titles)):
	plt.subplot(1, 2, i+1)
	plt.imshow(images[i], 'gray')
	plt.title(titles[i])
	plt.xticks([])
	plt.yticks([])
plt.show()