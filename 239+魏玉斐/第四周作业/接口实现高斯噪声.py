import cv2
import numpy as np
import skimage.util as util

# 读取图片
img = cv2.imread('lenna.png')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 实现高斯噪声
gaussian_noise = util.random_noise(img, mode='gaussian', rng=None, clip=True, mean=2, var=4)
# 实现椒盐噪声
salt_and_pepper_noise = util.random_noise(img, mode='s&p', amount=0.1, salt_vs_pepper=0.5, clip=True)

cv2.imshow("original", np.hstack([img, img1]))
# 显示原始图片和添加高斯噪声后的图片
# 水平排布原图和高斯噪声图
cv2.imshow(" gaussian_noise and salt", np.hstack([gaussian_noise, salt_and_pepper_noise]))

cv2.waitKey(0)
cv2.destroyAllWindows()
