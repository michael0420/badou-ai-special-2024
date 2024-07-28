import numpy as np
import matplotlib.pyplot as plt


def gaussian_smooth(image, kernel_size, sigma):
    '''使用高斯卷积核平滑图像'''
    # 创建高斯卷积核
    kernel = np.zeros((kernel_size, kernel_size))
    mean = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = np.exp(-((i - mean) ** 2 + (j - mean) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    # 归一化核
    kernel /= np.sum(kernel)

    # 边缘填充
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)),
                          mode='constant', constant_values=0)

    # 应用卷积核
    smoothed_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            smoothed_image[i, j] = np.sum(padded_image[i:i + kernel_size, j:j + kernel_size] * kernel)

    return smoothed_image


# 示例图像
image = np.random.rand(100, 100)  # 随机生成一个100x100的图像

# 高斯平滑
smoothed_image = gaussian_smooth(image, kernel_size=5, sigma=1)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(smoothed_image, cmap='gray')
plt.title('Smoothed Image'), plt.xticks([]), plt.yticks([])
plt.show()
