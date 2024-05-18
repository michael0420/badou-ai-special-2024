import cv2
import numpy as np
import random
from skimage import util


def salt_and_pepper_noise(src, percentage):
    # 确保输入图像是浮点数类型，以便进行数学运算
    rows, cols = src.shape

    # 计算要添加噪声的像素数量
    noise_num = int(percentage * rows * cols)

    # 椒盐噪声数量各占一半
    salt_num = noise_num // 2
    pepper_num = noise_num - salt_num

    # 生成随机索引以添加噪声
    salt_indices = [(random.randrange(rows), random.randrange(cols)) for _ in range(salt_num)]
    pepper_indices = [(random.randrange(rows), random.randrange(cols)) for _ in range(pepper_num)]

    # 生成相同大小的噪声数组并应用噪声
    for idx in salt_indices:
        src[idx] = 0

    # 应用盐噪声（255值）
    for idx in pepper_indices:
        src[idx] = 255

    return src


if __name__ == '__main__':
    # 读取图像
    img_source1 = cv2.imread('img.jpg', 0)  # 假设这是原图像
    img_source2 = cv2.imread('img.jpg', 0)  # 假设这是要添加噪声的图像
    img_source3 = cv2.imread('img.jpg', 0)  # 假设这是要添加噪声的图像

    # image_normalized = image.astype(np.float32) / 255.0  # 归一化，更方便

    # 添加高斯噪声
    img_s_and_p1 = salt_and_pepper_noise(img_source2, 0.05)
    img_s_and_p2 = util.random_noise(img_source3, mode="s&p")

    # 水平拼接图像
    # combined_image = np.hstack((img_source1 / 255.0, img_s_and_p))
    combined_image = np.hstack((img_source1 / 255.0, img_s_and_p1 / 255.0, img_s_and_p2))

    # 显示合并后的图像
    cv2.imshow('Original and Gaussian Noise Image', combined_image)
    cv2.waitKey(0)

    # 销毁所有窗口
    cv2.destroyAllWindows()
