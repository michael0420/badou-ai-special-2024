import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./lenna.png", 0)
print(img.shape)

rows, cols = img.shape

data = img.reshape((rows * cols, 1))
data = np.float32(data)
print(data)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

compact, labels, centers = cv2.kmeans(data, 32, None, criteria, 10, flags)
print(compact)
print(labels)
print(centers)

dst = labels.reshape((img.shape[0], img.shape[1]))

plt.rcParams['font.sans-serif'] = ['SimHei']

titles = ['原图', '聚类']
images = [img, dst]
for i in range(len(images)):
    plt.subplot(1, 2, i + 1)
    plt.imshow(images[i], "gray")
    plt.title(titles[i])
    # plt.xticks([])
    # plt.yticks([])
plt.show()
