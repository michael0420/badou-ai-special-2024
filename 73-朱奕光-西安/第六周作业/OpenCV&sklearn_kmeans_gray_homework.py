import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('lenna.png', 0)

"""
用sklearn库中KMeans函数实现
"""
img_sk = img.reshape([img.shape[0] * img.shape[1], 1])
clt = KMeans(n_clusters=4)
data_sk = clt.fit_predict(img_sk)
# print(data_sk)
data_sk = [[n] for n in data_sk]
# print(data_sk)
data_sk = np.asarray(data_sk)
# print(data_sk)
# print(data_sk.shape)
dst_sk = data_sk.reshape([img.shape[0], img.shape[1]])
print(dst_sk)
dstCvshowSk = dst_sk.astype(float)
dstCvshowSk = (255 * (dstCvshowSk - dst_sk.min()) / (dst_sk.max() - dst_sk.min())).astype(
    np.uint8)  # 如需cv2.imshow显示，需进行像素值归一化处理，并把数据转换成unit8
plt.subplot(121)
plt.title('sklearn')
plt.imshow(dst_sk, cmap='gray')
cv2.imshow('sklearn', dstCvshowSk)

"""
用OpenCV中cv2.kmeans函数实现
"""
img_cv=img.reshape(img.shape[0] * img.shape[1], 1).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
a, data_cv, b = cv2.kmeans(data=img_cv, K=4, bestLabels=None, criteria=criteria, attempts=10, flags=flags, centers=None)
dst_cv = data_cv.reshape(img.shape[0], img.shape[1])
plt.subplot(122)
plt.title('OpenCV')
plt.imshow(dst_cv, cmap='gray')
plt.show()
dstCvshowCv = dst_cv.astype(float)
dstCvshowCv = (255 * (dstCvshowCv - dst_cv.min()) / (dst_cv.max() - dst_cv.min())).astype(np.uint8)  # 归一化
cv2.imshow('OpenCv', dstCvshowCv)
cv2.waitKey(0)
cv2.destroyAllWindows()