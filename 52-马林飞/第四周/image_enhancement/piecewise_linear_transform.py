import cv2
import numpy as np


def piecewise_linear_transform(img):
    # 定义分阶段线性变换的阶段边界和对应的线性变换参数
    boundaries = [50, 100, 150, 200]
    alphas = [1.0, 1.5, 0.8, 0.5]
    betas = [0, 20, -30, 50]

    # 对图像的每个像素值应用分阶段线性变换
    transformed_img = np.zeros_like(img)
    for i in range(len(boundaries)):
        lower_bound = boundaries[i - 1] if i > 0 else 0
        upper_bound = boundaries[i]
        mask = np.logical_and(img >= lower_bound, img <= upper_bound)
        transformed_img[mask] = alphas[i] * img[mask] + betas[i]

    return transformed_img


# 读取图像
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# 分阶段线性变换
enhanced_img = piecewise_linear_transform(img)

# 显示原始图像和增强后的图像
cv2.imshow('Original Image', img)
cv2.imshow('Enhanced Image', enhanced_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
