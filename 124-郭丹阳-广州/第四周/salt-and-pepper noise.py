import cv2
import numpy as np

# 读取图像
image_path = 'D:/AI--pan/code/lenna.png'
image = cv2.imread(image_path)

# 添加椒盐噪声
noise_ratio = 0.02  # 噪声比例，即在图像中随机选择的像素占总像素数量的比例
amount = 0.5  # 椒盐噪声的数量级

# 获取图像的形状（高度，宽度，通道数）
height, width, channels = image.shape

# 计算要添加噪声的像素数量
num_noise_pixels = int(noise_ratio * height * width)

# 在图像中随机选择要添加椒盐噪声的像素位置
for _ in range(num_noise_pixels):
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    
    # 随机选择要添加椒盐噪声的像素值
    noise_value = np.random.choice([0, 255])  # 0代表黑色（椒），255代表白色（盐）
    
    # 将选定位置的像素值设置为椒盐噪声的像素值
    image[y, x] = [noise_value] * channels

# 显示图像
cv2.imshow('Noisy Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
