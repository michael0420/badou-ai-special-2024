import cv2
import matplotlib.pyplot as plt

# 读取图像
gray_image = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

# 应用Sobel算子
x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # ksize=3表示卷积核大小=3
y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

gradient_magnitude = cv2.magnitude(x, y)
gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

# 显示原始图像和边缘检测结果
plt.figure(figsize=(10, 5))

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')

# 显示边缘检测结果
plt.subplot(1, 2, 2)
plt.imshow(gradient_magnitude_normalized, cmap='gray')
plt.title('Sobel Edge Detection')

plt.show()
