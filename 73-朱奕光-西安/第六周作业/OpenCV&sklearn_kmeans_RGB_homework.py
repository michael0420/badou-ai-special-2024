import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('lenna.png')

"""
sklearn实现
"""
img_src_sk = img.reshape([-1, 3])
# print(img_src.shape)
kmeans = KMeans(n_clusters=4)
data_sk = kmeans.fit_predict(img_src_sk)
# print(data.shape)
data_match_sk = kmeans.cluster_centers_[kmeans.labels_]
# print(data_compass.shape)
dst_sk = data_match_sk.reshape(img.shape)
dst_sk = dst_sk.astype('uint8')
# print(dst.shape)
cv2.imshow('sklearn K=4', dst_sk)
dst_sk = cv2.cvtColor(dst_sk, cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.title('sklearn K=4')
plt.imshow(dst_sk, cmap='gray')

"""
OpenCV实现
"""
img_src_cv = img.reshape([-1, 3])
# print(img_src_cv.dtype)
data_cv = img_src_cv.astype(np.float32)
# print(data_cv.dtype)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
compactness, labels, centers = cv2.kmeans(data=data_cv, K=6, bestLabels=None, criteria=criteria, attempts=10,
                                          flags=cv2.KMEANS_RANDOM_CENTERS,
                                          centers=None)
labels=labels.flatten()
# labels = [n[0] for n in labels]
dst_cv=centers[labels]
dst_cv=dst_cv.reshape(img.shape).astype(np.uint8)
# print(dst_cv.shape)
# print(dst_cv.dtype)
cv2.imshow('OpenCV K=6',dst_cv)
cv2.waitKey(0)
dst_cv=cv2.cvtColor(dst_cv,cv2.COLOR_BGR2RGB)
plt.subplot(122)
plt.title('OpenCV K=6')
plt.imshow(dst_cv)
plt.show()