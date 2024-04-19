import cv2
import numpy as np

def bilinear_interpolation(image, new_size):
    # 获取图像尺寸
    height, width = image.shape[:2]
    new_height, new_width = new_size
    
    # 计算缩放比例
    scale_x = width / new_width
    scale_y = height / new_height
    
    # 创建一个新图像
    interpolated_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    for y in range(new_height):
        for x in range(new_width):
            # 计算在原始图像中的坐标
            x_original = x * scale_x
            y_original = y * scale_y
            
            # 计算在原始图像中最接近的四个像素的坐标
            x1 = int(x_original)
            y1 = int(y_original)
            x2 = min(x1 + 1, width - 1)
            y2 = min(y1 + 1, height - 1)
            
            # 计算双线性插值
            x_diff = x_original - x1
            y_diff = y_original - y1
            interpolated_image[y, x] = (
                image[y1, x1] * (1 - x_diff) * (1 - y_diff) +
                image[y1, x2] * x_diff * (1 - y_diff) +
                image[y2, x1] * (1 - x_diff) * y_diff +
                image[y2, x2] * x_diff * y_diff
            ).astype(np.uint8)
    
    return interpolated_image

# 加载图像
img = cv2.imread('D:/AI--pan/code/lenna.png')

# 指定新的图像尺寸
new_size = (800, 800)

# 执行双线性插值
interpolated_img = bilinear_interpolation(img, new_size)

# 显示原始图像和插值后的图像
cv2.imshow('Original Image', img)
cv2.imshow('Interpolated Image', interpolated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
