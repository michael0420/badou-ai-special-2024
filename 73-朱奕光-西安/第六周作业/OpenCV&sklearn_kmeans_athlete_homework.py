import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

A = np.float32([[0.0888, 0.5885],
                [0.1399, 0.8291],
                [0.0747, 0.4974],
                [0.0983, 0.5772],
                [0.1276, 0.5703],
                [0.1671, 0.5835],
                [0.1306, 0.5276],
                [0.1061, 0.5523],
                [0.2446, 0.4007],
                [0.1670, 0.4770],
                [0.2485, 0.4313],
                [0.1227, 0.4909],
                [0.1240, 0.5668],
                [0.1461, 0.5113],
                [0.2315, 0.3788],
                [0.0494, 0.5590],
                [0.1107, 0.4799],
                [0.1121, 0.5735],
                [0.1007, 0.6318],
                [0.2567, 0.4326],
                [0.1956, 0.4280]])

"""
运用sklearn中的KMeans函数
"""
i = KMeans(n_clusters=3)
dst_sk = i.fit_predict(A)
print('sklearn')
print(dst_sk)

plt.subplot(121)
plt.title("sklearn_kmeans")
plt.scatter([n[0] for n in A], [n[1] for n in A], c=dst_sk)
plt.xlabel('points')
plt.ylabel('assists')

"""
运用cv2.kmeans函数
"""
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
flags = cv2.KMEANS_RANDOM_CENTERS
retval, labels, centers = cv2.kmeans(data=A, K=3, bestLabels=None, criteria=criteria, attempts=10, flags=flags,
                                     centers=None)
index = [n[0] for n in labels]
# print(labels)
# print(labels.shape)
# print(type(labels))

plt.subplot(122)
plt.title("opencv_kmeans")
plt.scatter([n[0] for n in A], [n[1] for n in A], c=index)
plt.xlabel('points')
plt.ylabel('assists')
plt.show()      # 不同函数达成相同效果
