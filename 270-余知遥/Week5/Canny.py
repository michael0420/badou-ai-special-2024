import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_blur(image, kernel_size=5, sigma=1.4):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def gradient_intensity_and_direction(image):
    # 计算梯度幅值
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = cv2.filter2D(image, -1, Kx)
    Iy = cv2.filter2D(image, -1, Ky)
    G = np.hypot(Ix, Iy)
    theta = np.arctan2(Iy, Ix)
    return G, theta

def non_maximum_suppression(G, theta):
    # 非极大值抑制
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255
            # Angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = G[i, j + 1]
                r = G[i, j - 1]
            # Angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = G[i + 1, j - 1]
                r = G[i - 1, j + 1]
            # Angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                q = G[i + 1, j]
                r = G[i - 1, j]
            # Angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = G[i - 1, j - 1]
                r = G[i + 1, j + 1]

            if G[i, j] >= q and G[i, j] >= r:
                Z[i, j] = G[i, j]
            else:
                Z[i, j] = 0
    return Z

def thresholding(image, low_threshold, high_threshold):
    # 双阈值检测 强边缘与弱边缘
    high = image.max() * high_threshold
    low = high * low_threshold
    M, N = image.shape
    res = np.zeros((M, N), dtype=np.int32)
    weak = np.int32(25)
    strong = np.int32(255)
    strong_i, strong_j = np.where(image >= high)
    zeros_i, zeros_j = np.where(image < low)
    weak_i, weak_j = np.where((image <= high) & (image >= low))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong

def edge_tracking_by_hysteresis(image, weak, strong=255):
    # 边缘跟踪
    M, N = image.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if image[i, j] == weak:
                if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (image[i + 1, j + 1] == strong)
                        or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                        or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (
                                image[i - 1, j + 1] == strong)):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

def canny_edge_detection(image, low_threshold=0.05, high_threshold=0.15):
    # 高斯滤波
    blurred_image = gaussian_blur(image)
    # 计算梯度幅值
    G, theta = gradient_intensity_and_direction(blurred_image)
    # 应用非极大值抑制
    suppressed_image = non_maximum_suppression(G, theta)
    # 应用双阈值检测，强边缘与弱边缘
    thresholded_image, weak, strong = thresholding(suppressed_image, low_threshold, high_threshold)
    # 边缘跟踪
    final_edges = edge_tracking_by_hysteresis(thresholded_image, weak, strong)

    return final_edges

# 读取图像
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)
# 使用Canny算法检测边缘
edges = canny_edge_detection(image)
# 显示原始图像和边缘检测结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
