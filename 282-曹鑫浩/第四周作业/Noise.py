import cv2
import matplotlib.pyplot as plt
from skimage import util

src = cv2.imread('lenna.png', 1)
dst = util.random_noise(src, mode='poisson')

cv2.imshow('poisson_noise', dst)
cv2.waitKey()
cv2.destroyAllWindows()

