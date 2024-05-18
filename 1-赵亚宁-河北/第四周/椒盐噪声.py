
import numpy as np
import cv2


def salt_pepper_noise(image, s_vs_p=0.5, amount=0.8):
    """
    向图片添加椒盐噪声。

    :param image: 原始图片。
    :param s_vs_p: 椒盐比例。
    :param amount: 噪声的量。
    :return: 添加噪声后的图片。
    """
    output = image
    noise_type = np.random.randint(0, 2)
    if noise_type == 0:  # 添加椒盐噪声
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        output[coords] = 255

    else:  # 添加盐噪声
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        output[coords] = 0

    return output


# 读取图片

image=cv2.imread('lenna.png',0)
cv2.imshow('Original Image', image)
# 添加椒盐噪声
noisy_image = salt_pepper_noise(image, amount=0.8)
# 显示图片
cv2.imshow('Noisy Image', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()