import math

import cv2
import numpy as np
from sklearn import datasets
import cv2


class Lesson04:
    _source_img = []

    def __init__(self, source_img):
        if len(source_img) < 1:
            raise ValueError("source_img can`t be empty!")
        self._source_img = source_img

    def gaussian_distribution(self, mean, sigma):
        """给图片加上高斯噪声"""
        # 将像素按比例缩小，便于添加高斯噪声
        tmp_img = np.array(self._source_img / 255, dtype=float)
        # 生成高斯噪声数据
        noise = np.random.normal(loc=mean, scale=sigma, size=self._source_img.shape)
        # 给图像添加高斯噪声
        img_noise = tmp_img + noise
        # 图像存储的数据类型为uint8，不转则无法输出图像,且将书按原比例放大
        return np.uint8(img_noise * 255)

    def salt_pepper_noise(self, snr):
        """给图片加上椒盐噪声"""
        if snr < 0 or snr >= 1:
            raise ValueError("snr must be in (0,1)")
        height, width = self._source_img.shape
        # 转换数据类型，提升计算精度
        tmp_img = np.array(self._source_img, dtype=float)
        # 根据信噪比计算总噪点数
        noise_point_count = height * width * snr
        # 随机给像素点添加椒盐噪声
        for i in range(int(noise_point_count)):
            y = np.random.randint(height, size=1)
            x = np.random.randint(width, size=1)
            tag = np.random.choice(2, size=1)
            if tag == 0:
                tmp_img[x, y] = 0
            else:
                tmp_img[x, y] = 255
        return np.uint8(tmp_img)

    def linear_filter(self, kernel, mode='mean'):
        """中值滤波对椒盐噪声效果较好，均值滤波对高斯噪声、均匀随机分布噪声效果较好
            最大值、最小值滤波感觉对两种噪声都没效果
            mean：均值滤波
            median：中值滤波
            max-min：最大值、最小值滤波
        """
        height, width = kernel
        tmp_img = np.array(self._source_img, dtype=float)
        img_filter = np.zeros(self._source_img.shape)
        # 边缘填充是输入输出图像大小一致
        padding = (height - 1, width - 1)
        img_pad = np.pad(tmp_img, padding)
        for i in range(img_filter.shape[0]):
            for j in range(img_filter.shape[1]):
                if mode == 'mean':
                    # 计算均值
                    img_filter[i, j] = np.mean(img_pad[i: i + height, j: j + width])
                elif mode == 'median':
                    # 计算中值
                    img_filter[i, j] = np.median(img_pad[i: i + height, j: j + width])
                elif mode == 'max-min':
                    tmp = img_pad[i: i + height, j: j + width]
                    max_data = np.max(tmp)
                    min_data = np.min(tmp)
                    median = tmp[math.floor(height / 2), math.floor(width / 2)]
                    # print(f"max{max_data}")
                    # print(f"mix{min_data}")
                    # print(f"middle{median}")
                    if median < min_data:
                        img_filter[i, j] = min_data
                        print(f"替换为最小值{min_data}")
                    elif median > max_data:
                        img_filter[i, j] = max_data
                        print(f"替换为最大值{max_data}")
                    else:
                        img_filter[i, j] = img[i, j]
                        # print(f"值不变{img[i, j]}")

        return np.uint8(img_filter)


def pca_reduce(k=2):
    """pca算法"""
    dataset = datasets.load_iris()
    data = dataset.data
    # 0中心化
    mean = np.mean(data)
    data_mean = data - mean
    # 计算协方差矩阵
    data_d = np.matmul(data_mean.T, data_mean) / data.shape[0]
    # 计算特征值及特征向量
    eigen_values, eigen_vectors = np.linalg.eig(data_d)
    # 将旧数据映射到新的特征空间，生成新数据
    new_data = np.matmul(data, eigen_vectors[:, :2])
    return new_data


if __name__ == '__main__':
    # img = cv2.imread("image/lenna.png", cv2.IMREAD_GRAYSCALE)
    # lesson = Lesson04(img)
    # 添加高斯噪声
    # result = lesson.gaussian_distribution(0.0, 0.1)
    # cv2.imshow("gaussian noise", result)

    # 对高斯噪声进行均值滤波
    # gaussian_noise = Lesson04(result)
    # result2 = gaussian_noise.linear_filter(kernel=(3, 3), mode='mean')
    # cv2.imshow("linear mean filter", result2)

    # 添加椒盐噪声
    # result3 = lesson.salt_pepper_noise(0.05)
    # cv2.imshow("salt-pepper noise", result3)
    # 对椒盐噪声进行中值滤波
    # salt_pepper_noise_img = Lesson04(result3)
    # result4 = salt_pepper_noise_img.linear_filter(kernel=(3, 3), mode="median")
    # cv2.imshow("linear median filter", result4)

    # cv2.waitKey(0)
    print(pca_reduce(2))