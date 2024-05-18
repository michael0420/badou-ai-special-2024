import cv2
import numpy as np

# 读取原始图像
img = cv2.imread('lenna.png')

# 图像二维像素转换为一维
data = img.reshape((-1,3))
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# 创建窗口
cv2.namedWindow('image')

# 调节杆回调函数
def update(val):
    # K-Means聚类
    compactness, labels, centers = cv2.kmeans(data, val, None, criteria, 10, flags)

    # 图像转换回uint8二维类型
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    dst = res.reshape((img.shape))

    # 显示图像
    cv2.imshow('image', dst)

# 创建调节杆
cv2.createTrackbar('K', 'image', 2, 64, update)

# 初始化显示
update(2)

cv2.waitKey(0)
cv2.destroyAllWindows()