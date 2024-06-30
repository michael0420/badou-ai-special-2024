# 1.实现高斯噪声 2.实现椒盐噪声 3.实现PCA
import random

import cv2
import numpy as np
from tqdm import tqdm

def createNoise(img, percentage, noise_type, means=0, sigma=0):
    image = img.copy()
    h,w = image.shape
    noise_num = int(percentage*h*w)

    for i in tqdm(range(noise_num)):
        randX = np.random.randint(0, w)
        randY = np.random.randint(0, h)
        if noise_type == "Gaussian":
            image[randY,randX] = image[randY,randX]+random.gauss(means, sigma)
        if noise_type == "SpicySalt":
            if random.random() <= 0.5:
                image[randY,randX] = 0
            else:
                image[randY,randX] = 255
        if image[randY,randX]<0:
            image[randY,randX] = 0
        elif image[randY,randX]>255:
            image[randY,randX] = 255

    return image


# img = cv2.imread("../lenna.png")
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gussian_img = createNoise(gray_img, 0.8, "Gaussian", 5, 10)
# spicy_salt_img = createNoise(gray_img, 0.1, "SpicySalt")
# cv2.imshow('gray_img',gray_img)
# cv2.imshow('gussian_img',gussian_img)
# cv2.imshow('spicy_salt_img',spicy_salt_img)
# cv2.waitKey(0)

########################主成分分析##################

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]

        #去中心化
        X = X - X.mean(axis=0)
        # 求协方差矩阵
        self.covariance = np.dot(X.T, X) / X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        # 对X进行降维
        return np.dot(X, self.components_)


# 调用
pca = PCA(n_components=2)
X = np.array(
    [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
newX = pca.fit_transform(X)
print(newX)