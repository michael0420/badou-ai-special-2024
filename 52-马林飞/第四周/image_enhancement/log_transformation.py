import cv2
import numpy as np

# 读取图像
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# 对数变换
c = 1  # 对数变换的参数，可以根据需要调整
log_transformed_img = c * np.log(1 + img)

# 将浮点数转换为整数类型，方便显示
log_transformed_img = np.uint8(log_transformed_img)

# 显示原始图像和对数变换后的图像
cv2.imshow('Original Image', img)
cv2.imshow('Log Transformed Image', log_transformed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
