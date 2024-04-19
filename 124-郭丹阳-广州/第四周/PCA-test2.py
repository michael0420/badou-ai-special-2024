import cv2
import numpy as np

# 读取图像
image_path = 'D:/AI--pan/code/lenna.png'
image = cv2.imread(image_path)

# 将图像转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 将灰度图像转换为二维数组
rows, cols = gray.shape
data = gray.reshape(rows * cols, 1).astype(np.float32)

# 计算PCA
mean, eigenvectors = cv2.PCACompute(data, mean=None)

# 选择前N个特征向量（N个主成分）
num_components = 3  # 选择前3个主成分
eigenvectors = eigenvectors[:num_components]

# 通过投影计算低维表示
transformed_data = np.dot(data - mean, eigenvectors.T)

# 将低维数据转换回原始维度
reconstructed_data = np.dot(transformed_data, eigenvectors) + mean
reconstructed_data = reconstructed_data.reshape(rows, cols).astype(np.uint8)

# 显示原始图像和重建图像
cv2.imshow('Original Image', gray)
cv2.imshow('Reconstructed Image', reconstructed_data)
cv2.waitKey(0)
cv2.destroyAllWindows()
