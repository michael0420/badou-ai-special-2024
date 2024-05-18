"""invocation interface Noise
来自skimage里面带有的模块util
"""
import cv2
from skimage import util
img = cv2.imread('E:/Desktop/jianli/lenna.png')
dst = util.random_noise(img, mode='s&p')
cv2.imshow('Source', img)
cv2.imshow('Noise', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
