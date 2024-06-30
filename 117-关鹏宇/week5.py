# 1.实现canny_detail  2.实现透视变换


import cv2
import numpy as np

#透视变换
def PerspectTrans(image, src, dst):
    m = cv2.getPerspectiveTransform(src, dst)
    print("warpMatrix:")
    print(m)
    result = cv2.warpPerspective(image, m, (337, 488))
    cv2.imshow("src", img)
    cv2.imshow("result", result)
    return result

#canny边缘检测
def gaussian_kernel(size, sigma=1):
    """生成高斯核"""
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def gaussian_blur(image, kernel_size=5, sigma=1):
    """高斯模糊"""
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

def sobel_edge_detection(image):
    """使用Sobel算子计算梯度"""
    ksize = 3
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    # 计算梯度幅度和方向
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x) * 180. / np.pi
    return gradient_magnitude, gradient_direction

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    """非极大值抑制"""
    height, width = gradient_magnitude.shape
    suppressed_gradient = np.zeros_like(gradient_magnitude)

    for x in range(1, width-1):
        for y in range(1, height-1):
            direction = gradient_direction[y, x]
            if direction < -22.5:
                direction = 180 + direction
            elif direction > 22.5 and direction <= 67.5:
                direction = direction - 22.5
            elif direction > 67.5 and direction <= 112.5:
                direction = direction - 67.5
            elif direction > 112.5 and direction <= 157.5:
                direction = direction - 112.5

            # 根据方向决定比较的像素
            if 0 <= direction < 22.5 or 157.5 <= direction <= 180:
                if gradient_magnitude[y, x] >= gradient_magnitude[y, x+1] and gradient_magnitude[y, x] >= gradient_magnitude[y, x-1]:
                    suppressed_gradient[y, x] = gradient_magnitude[y, x]
            elif 22.5 <= direction < 67.5:
                if gradient_magnitude[y, x] >= gradient_magnitude[y-1, x+1] and gradient_magnitude[y, x] >= gradient_magnitude[y+1, x-1]:
                    suppressed_gradient[y, x] = gradient_magnitude[y, x]
            elif 67.5 <= direction < 112.5:
                if gradient_magnitude[y, x] >= gradient_magnitude[y-1, x] and gradient_magnitude[y, x] >= gradient_magnitude[y+1, x]:
                    suppressed_gradient[y, x] = gradient_magnitude[y, x]
            elif 112.5 <= direction < 157.5:
                if gradient_magnitude[y, x] >= gradient_magnitude[y+1, x+1] and gradient_magnitude[y, x] >= gradient_magnitude[y-1, x-1]:
                    suppressed_gradient[y, x] = gradient_magnitude[y, x]

    return suppressed_gradient
def double_threshold_and_hysteresis(suppressed_gradient, low_threshold_ratio=0.1, high_threshold_ratio=0.3):
    """双阈值检测与边缘连接"""
    high_threshold = np.max(suppressed_gradient) * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    strong_edges = suppressed_gradient >= high_threshold
    weak_edges = (suppressed_gradient >= low_threshold) & (suppressed_gradient < high_threshold)

    # 进行边缘连接
    edges = np.copy(strong_edges)
    for y in range(1, edges.shape[0]-1):
        for x in range(1, edges.shape[1]-1):
            if edges[y, x] == 0 and weak_edges[y, x] == 1:
                if np.any(edges[y-1:y+2, x-1:x+2]):
                    edges[y, x] = 1

    return edges.astype(np.uint8) * 255

def canny_edge(image, sigma=1, low_threshold_ratio=0.1, high_threshold_ratio=0.3):
    blurred = gaussian_blur(image, kernel_size=5, sigma=sigma)
    gradient_magnitude, gradient_direction = sobel_edge_detection(blurred)
    suppressed_gradient = non_maximum_suppression(gradient_magnitude, gradient_direction)
    edges = double_threshold_and_hysteresis(suppressed_gradient, low_threshold_ratio, high_threshold_ratio)
    return edges

# 加载图像并应用自定义的Canny边缘检测
image = cv2.imread('../lenna.png', cv2.IMREAD_GRAYSCALE)
edges = canny_edge(image)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)







# img = cv2.imread('../photo1.jpg')
# '''
# 注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
# '''
# src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
# dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# result = PerspectTrans(img, src, dst)
# cv2.imshow("src", img)
# cv2.imshow("result", result)
# cv2.waitKey(0)