import cv2
import numpy as np

if __name__ == '__main__':
    # 读取图像
    image = cv2.imread("../data/lenna.png", cv2.IMREAD_GRAYSCALE)

    # 使用 Sobel 算子计算 x 方向和 y 方向的梯度
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # 计算 x 方向梯度
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # 计算 y 方向梯度

    # 计算梯度幅值和方向
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    gradient_direction = np.arctan2(sobely, sobelx)

    # 将幅值归一化到 0-255 范围内
    gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 显示边缘检测结果
    cv2.imshow('Sobel Edge Detection', gradient_magnitude_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()