import cv2
import numpy as np
from PIL import Image
from skimage import util

img = cv2.imread("lenna.png")
img_noise = util.random_noise(img,mode="poisson")
cv2.imshow("source",img)
cv2.imshow("noise",img_noise)
cv2.waitKey()
