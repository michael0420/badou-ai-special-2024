import cv2
from matplotlib import pyplot as plt
import numpy as np

# 加载彩色图像
image = cv2.imread('R-C2.jpg', 1)

# 分离图像通道
channels = cv2.split(image)

# 原直方图
plt.figure()
plt.hist(channels[0].ravel(), 256, color='b', alpha=0.5)
plt.hist(channels[1].ravel(), 256, color='g', alpha=0.5)
plt.hist(channels[2].ravel(), 256, color='r', alpha=0.5)
plt.show()

dst_b = cv2.equalizeHist(channels[0])
dst_g = cv2.equalizeHist(channels[1])
dst_r = cv2.equalizeHist(channels[2])

dst = cv2.merge((dst_b, dst_g, dst_r))
# 均衡化后的直方图
plt.figure()
plt.hist(dst_b.ravel(), 256, color='b', alpha=0.5)
plt.hist(dst_g.ravel(), 256, color='g', alpha=0.5)
plt.hist(dst_r.ravel(), 256, color='r', alpha=0.5)
plt.show()


cv2.imshow("img", np.hstack([image, dst]))
cv2.waitKey(0)