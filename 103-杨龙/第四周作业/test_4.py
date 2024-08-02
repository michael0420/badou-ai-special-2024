# -*- coding:utf-8 -*-

__author__ = "Young"

"""
第四周作业
1.实现高斯噪声 
2.实现椒盐噪声 
3.实现PCA 
4.拓展：证明中心化协方差矩阵公式
"""

import cv2
import myimage
import numpy as np
import mymatrix
import PCAUtil

"""
1. 实现高斯噪声
"""
lenna = myimage.MyImage('../Image/lenna.png')
gaussian_noise_img = lenna.noise_image(0.8, mode='gaussian', means=2, sigma=4)
cv2.imshow('gaussian noise gray image', gaussian_noise_img)
cv2.imshow('lenna gray image', lenna.gray_image())
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
2. 实现椒盐噪声
"""
lenna = myimage.MyImage('../Image/lenna.png')
salt_img = lenna.noise_image(0.6, 'salt')
pepper_img = lenna.noise_image(0.5, 'pepper')
salt_pepper_img = lenna.noise_image(0.4, 'salt&pepper')
cv2.imshow('salt', salt_img)
cv2.imshow('pepper', pepper_img)
cv2.imshow('salt_pepper', salt_pepper_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
3. 实现PCA
"""
X = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9, 35],
              [42, 45, 11],
              [9, 48, 5],
              [11, 21, 14],
              [8, 5, 15],
              [11, 12, 21],
              [21, 20, 25]])
k = np.shape(X)[1] - 1
my_matrix = mymatrix.MyMatrix(X, k)
my_matrix.principal_component_analysis()
print('样本矩阵：\n', my_matrix.X)
print('中心化后的样本矩阵: \n', my_matrix.center_X)
print('样本矩阵的协方差矩阵：\n', my_matrix.covariance_X)
print('%d阶降维转换矩阵：\n' % my_matrix.k, my_matrix.transposition_Z)
print('样本矩阵X的降维矩阵Z：\n', my_matrix.Z)

# 3.1 iris  展示PCA后的鸢尾花图像
PCAUtil.show_iris()

# 3.2  numpy PCA， sklearn PCA
data = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
matrix_h = mymatrix.MyMatrix(data, 2)
matrix_numpy = mymatrix.MyMatrix(data, 2)
matrix_sklearn = mymatrix.MyMatrix(data, 2)
matrix_h.principal_component_analysis()
matrix_numpy.pca_numpy()
matrix_sklearn.pca_sklearn()
print('matrix_h:\n', matrix_h.Z)
print('matrix_numpy:\n', matrix_numpy.Z)
print('matrix_sklearn:\n', matrix_sklearn.Z)


"""
4 证明中心化协方差矩阵公式
求协方差的公式为：C = (1/m)∑(每项-平均值)² 且中心化的矩阵X的转置*X正好是求和项的值
=> C = (1/m)*X的转置*X  (X为中心化后的矩阵)
"""

