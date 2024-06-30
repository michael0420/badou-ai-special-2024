import cv2
import numpy as np

# 读取图像
image = cv2.imread('img.png')

# 将图像转换为一维数组，每个像素点一个颜色值
reshaped_image = image.reshape(-1, 3).astype(np.float32)

# 应用K-means聚类
'''
终止条件:
cv2.TERM_CRITERIA_EPS: 表示聚类中心的变化小于指定精度时终止
cv2.TERM_CRITERIA_MAX_ITER: 表示最大迭代次数
100表示最大迭代次数为100
0.1表示变化精度小于0.1
'''
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
k = 6  # 假设我们想要聚成5个颜色
ret, labels, centers = cv2.kmeans(reshaped_image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 将聚类中心转换为整数，并重新创建图像
centers = np.uint8(centers)
print(ret)  # ret
print(labels)  # 每个颜色对应的索引
print(labels.flatten())  # 二维转一维
print(centers)  # k个中心点
clustered_image = centers[labels.flatten()]

# 将一维数组转换回图像尺寸
clustered_image = clustered_image.reshape(image.shape)

# 显示原始图像和聚类后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Clustered Image', clustered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
