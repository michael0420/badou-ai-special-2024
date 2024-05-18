import cv2
import numpy
from skimage import util


img = cv2.imread("../../lenna.png")
img_gaussian = util.random_noise(img,mode='gaussian')

cv2.imshow("img",img)
cv2.imshow("img_gaussian",img_gaussian)

cv2.waitKey(0)









