import numpy as np
import cv2


def sobel_filter(image, k_size=3):
    des_img = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    # 边界扩充(padding)
    pad_size = k_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    # 转换为浮点类型
    image_float = padded_image.astype(np.float32)
    # 计算每个像素的Sobel值
    Gx = np.zeros_like(image).astype(np.float32)
    Gy = np.zeros_like(image).astype(np.float32)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # 提取局部区域
            region = image_float[y:y + k_size, x:x + k_size]
            # print(region)
            tmp_x = region * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            tmp_y = region * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            # 计算Gx和Gy
            Gx[y, x] = tmp_x.sum()
            Gy[y, x] = tmp_y.sum()
    # 合并Gx和Gy
    gradient = np.sqrt(Gx ** 2 + Gy ** 2)
    # 转换回8位整数
    des_img = np.absolute(gradient).astype(np.uint8)
    return des_img


# 读取图像
image = cv2.imread('lenna.png', 0)
# 应用Sobel滤波器
sobel_img = sobel_filter(image, k_size=3)

x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
y = cv2.Sobel(image, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# 显示结果
cv2.imshow("Sobel Result", np.hstack([sobel_img, absX, absY, dst]))
cv2.waitKey(0)
cv2.destroyAllWindows()