import numpy as np
from skimage import util
import cv2

img = cv2.imread('lenna.png')

# 高斯噪声
img_gaussian = util.random_noise(img, 'gaussian', mean=0.5, var=0.05)
cv2.imshow("source", img)
print(img)
cv2.imshow('gaussian', img_gaussian)
print(img_gaussian)
cv2.imshow('merge', np.hstack([img, img_gaussian]))  # 分别是整型和浮点型图像矩阵，所以原图此时不能显示出来

# 泊松噪声
img_poisson = util.random_noise(img, 'poisson')
cv2.imshow('poisson', img_poisson)

# 盐噪声 撒白点
img_salt = util.random_noise(img, 'salt')  # pepper椒噪声，撒黑点
cv2.imshow('salt', img_salt)
# print(img_salt)

# 椒盐噪声 撒黑白点
img_sp = util.random_noise(img, 's&p')
cv2.imshow('poisson', img_sp)
# print(img_sp)
cv2.waitKey()
