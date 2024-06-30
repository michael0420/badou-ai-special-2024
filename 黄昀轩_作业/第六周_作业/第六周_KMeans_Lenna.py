import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

img = cv2.imread("lenna.png")
print(img)
print(img.shape)

img_r = img.reshape((-1, 3))
data = np.float32(img_r)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)  # 标准类型
fig = plt.figure(figsize=(20, 10))

images = [img]
for i in range(5):
    # 聚类数量
    K = (i + 1) * 2
    # Kmeans 计算
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # 复原图像
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)
    images.append(res2)

print(len(images))
for n in range(6):
    plt.subplot(2, 3, n + 1)
    plt.imshow(images[n])
    plt.title((f"K={n * 2}"))
    xtick = ([0, 128, 256, 384, 512]),
    ytick = ([0, 128, 256, 384, 512])

plt.show()
