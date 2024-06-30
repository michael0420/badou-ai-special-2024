from skimage import util
import cv2

image = cv2.imread('../week5/lenna.png', 0)
image1 = util.random_noise(image,mode='poisson')

cv2.imshow('src',image)
cv2.imshow('poisson',image1)
cv2.waitKey(0)
cv2.destroyAllWindows()