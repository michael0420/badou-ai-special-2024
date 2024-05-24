import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
## cv2方式
img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换为灰度图

sift = cv2.SIFT_create() # 创建sift检测器
keypoints, descriptor = sift.detectAndCompute(gray, None)

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))
                        
#img=cv2.drawKeypoints(gray,keypoints,img)

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Numpy方式
def generate_gaussian_pyramid(image, num_octaves, num_scales, sigma):
    pyramid = []
    k = 2 ** (1 / num_scales)
    for octave in range(num_octaves):
        scales = []
        for scale in range(num_scales + 3):
            sigma_effective = sigma * (k ** scale)
            blurred = gaussian_filter(image, sigma_effective)
            scales.append(blurred)
        pyramid.append(scales)
        image = cv2.pyrDown(image)
    return pyramid

def generate_dog_pyramid(gaussian_pyramid):
    dog_pyramid = []
    for scales in gaussian_pyramid:
        dog_scales = []
        for i in range(1, len(scales)):
            dog = scales[i] - scales[i - 1]
            dog_scales.append(dog)
        dog_pyramid.append(dog_scales)
    return dog_pyramid

def find_keypoints(dog_pyramid, contrast_threshold=0.04):
    keypoints = []
    for octave_idx, dog_scales in enumerate(dog_pyramid): ## 遍历高斯差分金字塔
        for scale_idx in range(1, len(dog_scales) - 1): ## 遍历每个尺度
            for i in range(1, dog_scales[scale_idx].shape[0] - 1): ## 遍历x坐标
                for j in range(1, dog_scales[scale_idx].shape[1] - 1): ## 遍历y坐标
                    for scale_idx in range(1, len(dog_scales) - 1): ## 遍历每个尺度
                        for i in range(1, dog_scales[scale_idx].shape[0] - 1):
                            for j in range(1, dog_scales[scale_idx].shape[1] - 1):
                                patch = np.array([dog_scales[scale_idx - 1][i - 1:i + 2, j - 1:j + 2],
                                                  dog_scales[scale_idx][i - 1:i + 2, j - 1:j + 2],
                                                  dog_scales[scale_idx + 1][i - 1:i + 2, j - 1:j + 2]])
                    if is_extremum(patch, contrast_threshold):
                        keypoints.append((octave_idx, scale_idx, i, j))
    return keypoints

def is_extremum(patch, contrast_threshold):
    center_value = patch[1, 1, 1] ## 获取中心值
    if abs(center_value) > contrast_threshold: ## 判断中心值是否大于阈值
        if center_value == np.max(patch) or center_value == np.min(patch): ## 判断中心值是否是局部最大值或局部最小值
            return True
    return False

def assign_orientation(gaussian_pyramid, keypoints): ## 为关键点分配方向
    oriented_keypoints = [] ## 保存关键点的方向
    for (octave_idx, scale_idx, i, j) in keypoints: ## 遍历关键点
        image = gaussian_pyramid[octave_idx][scale_idx] ## 获取关键点所在的图像
        patch = image[i - 1:i + 2, j - 1:j + 2] ## 获取关键点周围的3x3邻域
        gradient_magnitude, gradient_orientation = compute_gradients(patch) ## 计算梯度幅度和方向
        hist, _ = np.histogram(gradient_orientation, bins=36, range=(-180, 180), weights=gradient_magnitude) ## 计算梯度方向直方图
        dominant_orientation = np.argmax(hist) * 10 - 180 ## 获取主方向
        oriented_keypoints.append((octave_idx, scale_idx, i, j, dominant_orientation)) ## 保存关键点的方向
    return oriented_keypoints

def compute_gradients(patch):
    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3) ## 计算x方向梯度
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3) ## 计算y方向梯度
    magnitude = np.sqrt(gx ** 2 + gy ** 2) ## 计算梯度幅度
    orientation = np.arctan2(gy, gx) * (180 / np.pi) ## 计算梯度方向
    return magnitude, orientation

def compute_sift_descriptors(gaussian_pyramid, oriented_keypoints): ## 计算SIFT描述子
    descriptors = []
    for (octave_idx, scale_idx, i, j, orientation) in oriented_keypoints: ## 遍历关键点
        image = gaussian_pyramid[octave_idx][scale_idx] ## 获取关键点所在的图像
        patch = image[i - 8:i + 8, j - 8:j + 8] ## 获取关键点周围的16x16邻域
        magnitude, direction = compute_gradients(patch) ## 计算梯度幅度和方向
        direction -= orientation ## 减去关键点的方向
        hist, _ = np.histogram(direction, bins=8, range=(-180, 180), weights=magnitude) ## 计算梯度方向直方图
        descriptors.append(hist.flatten()) ## 将直方图展平
    return np.array(descriptors)

# 加载图像并转为灰度图
image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# 生成高斯金字塔和高斯差分金字塔
num_octaves = 4
num_scales = 3
sigma = 1.6
gaussian_pyramid = generate_gaussian_pyramid(image, num_octaves, num_scales, sigma)
dog_pyramid = generate_dog_pyramid(gaussian_pyramid)

# 检测关键点并为其分配方向
keypoints = find_keypoints(dog_pyramid)
oriented_keypoints = assign_orientation(gaussian_pyramid, keypoints)

# 计算SIFT描述子
descriptors = compute_sift_descriptors(gaussian_pyramid, oriented_keypoints)

# 绘制关键点
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) ## 将灰度图转为BGR图
for (octave_idx, scale_idx, i, j, orientation) in oriented_keypoints: ## 遍历关键点
    cv2.circle(output_image, (j * (2 ** octave_idx), i * (2 ** octave_idx)), 2, (0, 255, 0), 1) ## 绘制关键点
cv2.imshow('SIFT Keypoints', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()