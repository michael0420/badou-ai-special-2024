import cv2
import random
import matplotlib.pyplot as plt

def GaussNoise(img, means, sigma, proportion):
    test_img = img.copy()  # 复制图像，避免修改原始图像
    h, w = test_img.shape[0:2]
    test_num = int(proportion * h * w)  # 计算需要添加噪声的像素数

    for i in range(test_num):
        randomX = random.randint(0, h - 1)  # 随机选择一个行索引
        randomY = random.randint(0, w - 1)  # 随机选择一个列索引

        # 在随机位置添加高斯噪声
        test_img[randomX, randomY] += random.gauss(means, sigma)

        # 限制像素值在 [0, 255] 范围内
        if test_img[randomX, randomY] > 255:
            test_img[randomX, randomY] = 255
        elif test_img[randomX, randomY] < 0:
            test_img[randomX, randomY] = 0

    return test_img

# 读取图像
img = cv2.imread("../lenna.png", 0)

# 添加高斯噪声
end_img = GaussNoise(img, 2, 4, 0.8)

# 使用 Matplotlib 显示图像
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')  # 图像标题
plt.axis('off')  # 关闭坐标轴

plt.subplot(1, 2, 2)
plt.imshow(end_img, cmap='gray')
plt.title('With Gaussian Noise')  # 图像标题
plt.axis('off')  # 关闭坐标轴

plt.show()  # 显示图像
