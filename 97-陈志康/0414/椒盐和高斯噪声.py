import numpy as np
import cv2
from skimage import util

# 获取图片
img = cv2.imread('../0327/lenna.png');
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化图片
# 调用噪声函数，实现噪声化,
gray_noise_img = util.random_noise(gray_img, 's&p')  # 加椒盐噪声
cv2.imshow('gray_img', gray_img)
cv2.imshow('gray_noise_img', gray_noise_img)

# 调用噪声函数，实现噪声化,
gaosi_noise_img = util.random_noise(img, 'gaussian')  #加高斯噪声
cv2.imshow('source', img)

cv2.imshow('gaosi_img', gaosi_noise_img)
cv2.waitKey(0)
# 销毁窗口，释放空间
cv2.destroyAllWindows();
